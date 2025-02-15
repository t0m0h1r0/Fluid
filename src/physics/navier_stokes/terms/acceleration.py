"""
密度勾配による加速度項を計算するモジュール

密度の空間的な変化による加速度項 -1/ρ u(u⋅∇ρ) を計算します。
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
        self, velocity: VectorField, density: ScalarField, **kwargs
    ) -> VectorField:
        """
        密度勾配による加速度項を計算

        Args:
            velocity: 速度場
            density: 密度場

        Returns:
            加速度項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 密度勾配の計算
        density_grad = [
            ScalarField(velocity.shape, velocity.dx, density.gradient(i))
            for i in range(velocity.ndim)
        ]

        # 速度と密度勾配の内積 (u⋅∇ρ)
        density_advection = ScalarField(velocity.shape, velocity.dx)
        for u_comp, grad_comp in zip(velocity.components, density_grad):
            density_advection = density_advection + (u_comp * grad_comp)

        # 結果の計算 -1/ρ u(u⋅∇ρ)
        result = VectorField(velocity.shape, velocity.dx)
        inv_density = ScalarField(density.shape, density.dx, 1.0) / (density + 1e-10)

        # 各成分の計算
        for i, v_comp in enumerate(velocity.components):
            result.components[i] = -(v_comp * density_advection * inv_density)

        # 診断情報の更新
        self._update_diagnostics(result, density_advection, density)

        return result

    def _update_diagnostics(
        self, result: VectorField, density_advection: ScalarField, density: ScalarField
    ):
        """
        診断情報を更新

        Args:
            result: 計算された加速度項
            density_advection: 密度の移流量
            density: 密度場
        """
        # 密度勾配の大きさ
        density_gradients = [
            np.max(np.abs(density.gradient(i))) for i in range(density.ndim)
        ]

        self._diagnostics = {
            "max_acceleration": float(
                max(np.max(np.abs(comp.data)) for comp in result.components)
            ),
            "density_advection": {
                "min": float(np.min(density_advection.data)),
                "max": float(np.max(density_advection.data)),
                "mean": float(np.mean(density_advection.data)),
            },
            "density_gradients": {
                "x": float(density_gradients[0]) if len(density_gradients) > 0 else 0.0,
                "y": float(density_gradients[1]) if len(density_gradients) > 1 else 0.0,
                "z": float(density_gradients[2]) if len(density_gradients) > 2 else 0.0,
            },
            "density_range": {
                "min": float(np.min(density.data)),
                "max": float(np.max(density.data)),
            },
        }

    def compute_timestep(
        self, velocity: VectorField, density: ScalarField, **kwargs
    ) -> float:
        """
        密度勾配による加速度項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場
            density: 密度場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 密度勾配の最大値を計算
        max_density_grad = max(
            np.max(np.abs(density.gradient(i))) for i in range(density.ndim)
        )

        # 最大速度の計算
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # 特性速度の推定: v_c ≈ max(|u|) * max(|∇ρ|/ρ)
        min_density = np.min(density.data)
        characteristic_speed = max_velocity * max_density_grad / (min_density + 1e-10)

        # CFL条件に基づく時間刻み幅の計算
        cfl = kwargs.get("cfl", 0.5)
        return cfl * min(velocity.dx) / (characteristic_speed + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag
