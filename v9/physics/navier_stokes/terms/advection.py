from typing import List, Dict, Any
import numpy as np

from core.field import VectorField
from .base import BaseNavierStokesTerm


class AdvectionTerm(BaseNavierStokesTerm):
    """移流項クラス"""

    def __init__(
        self,
        use_weno: bool = True,
        weno_order: int = 5,
        name: str = "Advection",
        enabled: bool = True,
    ):
        super().__init__(name, enabled)
        self._use_weno = use_weno
        self._weno_order = weno_order

    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        """移流項の寄与を計算"""
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        result = []
        for i, v_i in enumerate(velocity.components):
            flux = -sum(
                v_j.data * v_i.gradient(j) for j, v_j in enumerate(velocity.components)
            )
            result.append(flux)

        self._diagnostics["flux_max"] = float(max(np.max(np.abs(r)) for r in result))
        self._diagnostics["scheme"] = "WENO" if self._use_weno else "Central"

        return result

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """移流項に基づく時間刻み幅の制限を計算"""
        if not self.enabled:
            return float("inf")

        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        cfl = kwargs.get("cfl", 0.5)
        return cfl * velocity.dx / (max_velocity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "weno": {
                    "enabled": self._use_weno,
                    "order": self._weno_order if self._use_weno else None,
                }
            }
        )
        return diag
