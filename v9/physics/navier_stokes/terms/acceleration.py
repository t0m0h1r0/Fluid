"""
密度勾配による加速度項を計算するモジュール

密度の空間的な変化による加速度項を表現します。
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from .base import BaseNavierStokesTerm


class AccelerationTerm(BaseNavierStokesTerm):
    """
    密度勾配による加速度項を計算するクラス

    計算される項: -1/ρ u(u⋅∇ρ)
    """

    def __init__(
        self,
        name: str = "DensityGradientAcceleration",
        enabled: bool = True,
    ):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)

    def compute(
        self, 
        velocity: VectorField, 
        density: ScalarField,
        **kwargs
    ) -> VectorField:
        """
        密度勾配による加速度項を計算

        Args:
            velocity: 速度場
            density: 密度場

        Returns:
            密度勾配による加速度項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 結果用のVectorFieldを作成
        result = VectorField(velocity.shape, velocity.dx)
        dx = velocity.dx

        # 密度の勾配を計算
        density_gradients = [
            np.gradient(density.data, dx, axis=axis) 
            for axis in range(density.ndim)
        ]

        # 密度の勾配による加速度項を計算
        for i in range(velocity.ndim):
            # u⋅∇ρ の計算
            density_advection = sum(
                velocity.components[j].data * density_gradients[j]
                for j in range(velocity.ndim)
            )
            
            # -1/ρ u(u⋅∇ρ)
            result.components[i].data = (
                -velocity.components[i].data * density_advection / 
                np.maximum(density.data, 1e-10)  # ゼロ除算を防ぐ
            )

        # 診断情報の更新
        self._update_diagnostics(result)

        return result

    def _update_diagnostics(self, result: VectorField):
        """
        診断情報を更新

        Args:
            result: 計算された加速度項
        """
        self._diagnostics = {
            "max_acceleration": float(
                max(np.max(np.abs(comp.data)) for comp in result.components)
            ),
            "enabled": self.enabled,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """
        密度勾配による加速度項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 密度勾配による加速度の最大値を推定
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)
        cfl = kwargs.get("cfl", 0.5)
        
        return cfl * velocity.dx / (max_velocity + 1e-10)