"""
圧力ポアソン方程式の移流項の発散を計算するモジュール

Navier-Stokes方程式の移流項 u⋅∇u の発散 ∇⋅(u⋅∇u) を計算します。
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from .base import PoissonTerm


class AdvectionTerm(PoissonTerm):
    """
    圧力ポアソン方程式の移流項の発散を計算するクラス

    式: ∇⋅(u⋅∇u)
    
    この項は、速度場の非線形移流効果を表現し、 
    圧力ポアソン方程式の右辺に寄与します。
    """

    def __init__(
        self,
        name: str = "AdvectionDivergence",
        enabled: bool = True,
    ):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)

    def compute(
        self, velocity: VectorField, **kwargs
    ) -> ScalarField:
        """
        移流項の発散を計算 ∇⋅(u⋅∇u)

        Args:
            velocity: 速度場

        Returns:
            移流項の発散をScalarFieldとして返す
        """
        if not self.enabled:
            return ScalarField(velocity.shape, velocity.dx)

        # 移流項の発散を計算
        result = ScalarField(velocity.shape, velocity.dx)
        
        # u⋅∇u の各成分の発散を計算
        for i in range(velocity.ndim):
            # u_j * ∂u_i/∂x_j の発散
            advection_div = np.zeros_like(velocity.components[i].data)
            for j in range(velocity.ndim):
                # v_j * ∂u_i/∂x_j
                advection_term = velocity.components[j].data * velocity.components[i].gradient(j)
                # ∂/∂x_j(v_j * ∂u_i/∂x_j)
                advection_div += np.gradient(advection_term, velocity.dx[j], axis=j)
            
            # 結果に加算（移流項の発散は成分ごとに計算し、合計）
            result.data += advection_div

        # 診断情報の更新
        self._update_diagnostics(result, velocity)

        return result

    def _update_diagnostics(self, result: ScalarField, velocity: VectorField):
        """
        診断情報を更新

        Args:
            result: 計算された移流項の発散
            velocity: 速度場
        """
        self._diagnostics = {
            "max_advection_divergence": float(np.max(np.abs(result.data))),
            "velocity_max": {
                f"component_{i}": float(np.max(np.abs(comp.data)))
                for i, comp in enumerate(velocity.components)
            },
            "velocity_ranges": {
                f"component_{i}": {
                    "min": float(np.min(comp.data)),
                    "max": float(np.max(comp.data)),
                }
                for i, comp in enumerate(velocity.components)
            }
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag