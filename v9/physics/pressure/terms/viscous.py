"""
圧力ポアソン方程式の粘性項を計算するモジュール

粘性項 -∇⋅(∇ν⋅∇u) の計算を実装します。
"""

import numpy as np

from core.field import VectorField, ScalarField
from .base import PoissonTerm


class ViscousTerm(PoissonTerm):
    """粘性項の計算クラス"""

    def __init__(self, name: str = "Viscous", enabled: bool = True):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)

    def compute(
        self, velocity: VectorField, viscosity: ScalarField, **kwargs
    ) -> ScalarField:
        """
        粘性項 -∇⋅(∇ν⋅∇u) を計算

        Args:
            velocity: 速度場
            viscosity: 粘性場

        Returns:
            粘性項の発散を表すScalarField
        """
        if not self.enabled:
            return ScalarField(velocity.shape, velocity.dx)

        result = ScalarField(velocity.shape, velocity.dx)
        dx = velocity.dx

        # 各方向成分について
        for i in range(velocity.ndim):
            # ∇u_i を計算
            velocity_grad = np.zeros_like(velocity.components[i].data)

            for j in range(velocity.ndim):
                # ∂u_i/∂x_j を計算
                dui_dxj = velocity.components[i].gradient(j)

                # ∇ν⋅∇u_i の計算
                viscous_term = viscosity.gradient(j) * dui_dxj

                # 発散の計算
                velocity_grad += np.gradient(viscous_term, dx, axis=j)

            result.data -= velocity_grad  # 符号に注意

        # 診断情報の更新
        self._update_diagnostics(result, viscosity)

        return result

    def _update_diagnostics(self, result: ScalarField, viscosity: ScalarField):
        """
        診断情報を更新

        Args:
            result: 計算された粘性項
            viscosity: 粘性場
        """
        self._diagnostics = {
            "max_value": float(np.max(np.abs(result.data))),
            "min_value": float(np.min(result.data)),
            "mean_value": float(np.mean(result.data)),
            "norm": float(np.linalg.norm(result.data)),
            "viscosity_range": {
                "min": float(np.min(viscosity.data)),
                "max": float(np.max(viscosity.data)),
                "mean": float(np.mean(viscosity.data)),
            },
        }
