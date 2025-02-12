from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from .base import BaseNavierStokesTerm


class PressureTerm(BaseNavierStokesTerm):
    """圧力項クラス"""

    def __init__(
        self, density: float = 1000.0, name: str = "Pressure", enabled: bool = True
    ):
        """
        Args:
            density: 参照密度
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)
        self._density = density

    def compute(
        self, velocity: VectorField, pressure: ScalarField, **kwargs
    ) -> VectorField:
        """圧力項の寄与を計算

        Args:
            velocity: 速度場
            pressure: 圧力場

        Returns:
            各方向の速度成分への圧力項の寄与をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 結果用のVectorFieldを作成
        result = VectorField(velocity.shape, velocity.dx)
        dx = velocity.dx

        # 各方向の圧力勾配を計算
        for i in range(velocity.ndim):
            grad_p = np.gradient(pressure.data, dx, axis=i)
            result.components[i].data = -grad_p / self._density

        # 診断情報の更新
        self._diagnostics = {
            "pressure_min": float(np.min(pressure.data)),
            "pressure_max": float(np.max(pressure.data)),
            "pressure_mean": float(np.mean(pressure.data)),
            "density": self._density,
        }

        return result

    def compute_timestep(
        self, velocity: VectorField, pressure: ScalarField, **kwargs
    ) -> float:
        """圧力項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場
            pressure: 圧力場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 音速の粗い見積もり
        sound_speed = np.sqrt(np.max(pressure.data) / self._density)

        # CFL条件に基づく制限
        return 0.5 * velocity.dx / (sound_speed + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "pressure": {
                    "min": self._diagnostics.get("pressure_min", 0.0),
                    "max": self._diagnostics.get("pressure_max", 0.0),
                    "mean": self._diagnostics.get("pressure_mean", 0.0),
                },
                "density": self._density,
            }
        )
        return diag