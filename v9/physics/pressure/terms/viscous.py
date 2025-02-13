"""
粘性項（拡散項）の計算を提供するモジュール

Navier-Stokes方程式における粘性項 ∇⋅(ν∇u) を計算します。
"""

import numpy as np

from core.field import VectorField, ScalarField
from .base import PoissonTerm
from typing import Dict, Any


class ViscousTerm(PoissonTerm):
    """粘性項（拡散項）を計算するクラス"""

    def __init__(self, name: str = "Viscous", enabled: bool = True):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)

    def compute(
        self, velocity: VectorField, viscosity: ScalarField, **kwargs
    ) -> VectorField:
        """粘性項 ∇⋅(ν∇u) を計算

        Args:
            velocity: 速度場
            viscosity: 粘性係数場

        Returns:
            粘性項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        result = VectorField(velocity.shape, velocity.dx)

        # 各方向成分について
        for i, v_i in enumerate(velocity.components):
            viscous_term = np.zeros_like(v_i.data)
            for j in range(velocity.ndim):
                # ∂u_i/∂x_j を計算
                du_dx = v_i.gradient(j)
                # ν * ∂u_i/∂x_j を計算
                viscous_flux = viscosity.data * du_dx
                # ∂/∂x_j(ν * ∂u_i/∂x_j) を計算
                viscous_term += np.gradient(viscous_flux, velocity.dx[j], axis=j)

            result.components[i].data = viscous_term

        # 診断情報の更新
        self._update_diagnostics(result, viscosity)

        return result

    def _update_diagnostics(self, result: VectorField, viscosity: ScalarField):
        """診断情報を更新

        Args:
            result: 計算された粘性項
            viscosity: 粘性係数場
        """
        self._diagnostics = {
            "max_viscous": float(
                max(np.max(np.abs(comp.data)) for comp in result.components)
            ),
            "viscosity_range": {
                "min": float(np.min(viscosity.data)),
                "max": float(np.max(viscosity.data)),
                "mean": float(np.mean(viscosity.data)),
            },
            "component_max": {
                f"component_{i}": float(np.max(np.abs(comp.data)))
                for i, comp in enumerate(result.components)
            },
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag
