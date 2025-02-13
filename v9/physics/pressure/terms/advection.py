"""
圧力ポアソン方程式の移流項を計算するモジュール

移流項 ∇⋅(u⋅∇u) の計算を実装します。
"""

import numpy as np

from core.field import VectorField, ScalarField
from .base import PoissonTerm


class AdvectionTerm(PoissonTerm):
    """移流項の計算クラス"""

    def __init__(self, name: str = "Advection", enabled: bool = True):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)

    def compute(self, velocity: VectorField, **kwargs) -> ScalarField:
        """
        移流項 ∇⋅(u⋅∇u) を計算

        Args:
            velocity: 速度場

        Returns:
            移流項の発散を表すScalarField
        """
        if not self.enabled:
            return ScalarField(velocity.shape, velocity.dx)

        result = ScalarField(velocity.shape, velocity.dx)
        dx = velocity.dx

        # 各方向成分について
        for i in range(velocity.ndim):
            # u⋅∇u_i の計算
            advection = np.zeros_like(velocity.components[i].data)
            for j, v_j in enumerate(velocity.components):
                advection += v_j.data * velocity.components[i].gradient(j)

            # 発散の計算: ∂/∂x_i(u⋅∇u_i)
            result.data += np.gradient(advection, dx, axis=i)

        # 診断情報の更新
        self._update_diagnostics(result)

        return result

    def _update_diagnostics(self, result: ScalarField):
        """
        診断情報を更新

        Args:
            result: 計算された移流項
        """
        self._diagnostics = {
            "max_value": float(np.max(np.abs(result.data))),
            "min_value": float(np.min(result.data)),
            "mean_value": float(np.mean(result.data)),
            "norm": float(np.linalg.norm(result.data)),
        }
