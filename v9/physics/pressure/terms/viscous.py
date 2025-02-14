"""
粘性項（拡散項）の計算を提供するモジュール

Navier-Stokes方程式における粘性項 ∇⋅(μ∇²u) を計算します。
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from .base import PoissonTerm


class ViscousTerm(PoissonTerm):
    """粘性項（拡散項）を計算するクラス"""

    def __init__(self, name: str = "Viscous", enabled: bool = True):
        """粘性項計算器を初期化

        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)

    def compute(
        self, velocity: VectorField, viscosity: ScalarField, **kwargs
    ) -> ScalarField:
        """粘性項 ∇⋅(μ∇²u) を計算

        Args:
            velocity: 速度場
            viscosity: 粘性係数場

        Returns:
            粘性項の発散
        """
        if not self.enabled:
            return ScalarField(velocity.shape[:-1], velocity.dx)

        # 結果を格納するスカラー場
        result = ScalarField(velocity.shape[:-1], velocity.dx)

        # 空間次元数の取得
        ndim = len(velocity.shape) - 1  # VectorFieldの形状から最後の次元を除外

        # 各速度成分についてラプラシアンを計算
        for i in range(ndim):
            laplacian = np.zeros_like(velocity.components[i].data)
            
            # 各方向の2階微分を計算
            for j in range(ndim):
                # ∂²u_i/∂x_j²
                laplacian += np.gradient(
                    np.gradient(
                        velocity.components[i].data, 
                        velocity.dx[j], 
                        axis=j
                    ),
                    velocity.dx[j],
                    axis=j
                )

            # 粘性を考慮した拡散項の発散を計算
            viscous_flux = viscosity.data * laplacian
            div_viscous = np.zeros_like(viscous_flux)
            
            # 各方向の勾配を合計
            for j in range(ndim):
                div_viscous += np.gradient(viscous_flux, velocity.dx[j], axis=j)
            
            # 結果に加算
            result.data += div_viscous

        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "name": self.name,
            "enabled": self.enabled,
        }