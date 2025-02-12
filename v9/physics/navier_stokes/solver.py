"""
二相流のためのNavier-Stokes方程式ソルバー

運動方程式の各項を統合し、速度場の時間微分を計算します。
"""

from typing import Optional
import numpy as np

from core.field import VectorField, ScalarField
from .terms import (
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
    AccelerationTerm,
)


class NavierStokesSolver:
    """
    Navier-Stokes方程式のソルバー

    速度場、密度場、粘性場、圧力場から速度の時間微分を計算します。
    """

    def __init__(self):
        """ソルバーを初期化"""
        # 各項の初期化
        self.advection_term = AdvectionTerm()
        self.diffusion_term = DiffusionTerm()
        self.pressure_term = PressureTerm()
        self.acceleration_term = AccelerationTerm()

    def compute_velocity_derivative(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        pressure: ScalarField,
        force: Optional[VectorField] = None,
        **kwargs,
    ) -> VectorField:
        """
        速度の時間微分を計算

        式: ∂u/∂t = -u⋅∇u - 1/ρ u(u⋅∇ρ) - 1/ρ ∇p + 1/ρ ∇⋅(μ(∇u+∇uT)) + 1/ρ f

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            pressure: 圧力場
            force: 外力場（オプション）
            **kwargs: 追加のパラメータ

        Returns:
            速度の時間微分をVectorFieldとして返す
        """
        # 結果を格納するVectorFieldを作成
        result = VectorField(velocity.shape, velocity.dx)

        # 1. 移流項: -u⋅∇u
        advection = self.advection_term.compute(velocity)
        
        # 2. 密度勾配による加速度項: -1/ρ u(u⋅∇ρ)
        density_gradient = self.acceleration_term.compute(velocity, density)
        
        # 3. 粘性項: 1/ρ ∇⋅(μ(∇u+∇uT))
        diffusion = self.diffusion_term.compute(velocity, viscosity)
        
        # 4. 圧力項: -1/ρ ∇p
        pressure_grad = self.pressure_term.compute(velocity, pressure, density)
        
        # 5. 外力項: 1/ρ f
        if force is None:
            force = VectorField(velocity.shape, velocity.dx)

        # 各成分の時間微分を計算
        for i in range(velocity.ndim):
            result.components[i].data = (
                -advection.components[i].data      # 移流項
                + density_gradient.components[i].data  # 密度勾配項
                + diffusion.components[i].data     # 粘性項
                - pressure_grad.components[i].data # 圧力項
                + force.components[i].data / np.maximum(density.data, 1e-10)  # 外力項
            )

        # 診断情報の更新
        self._update_diagnostics(
            advection=advection,
            density_gradient=density_gradient,
            diffusion=diffusion,
            pressure_grad=pressure_grad,
            force=force,
            result=result,
        )

        return result

    def _update_diagnostics(self, **fields: VectorField) -> None:
        """
        診断情報を更新

        Args:
            **fields: 更新に使用する各VectorField
        """
        self._diagnostics = {}
        for name, field in fields.items():
            max_value = max(np.max(np.abs(comp.data)) for comp in field.components)