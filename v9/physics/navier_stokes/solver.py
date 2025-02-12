from typing import List, Dict, Any, Optional
import numpy as np

from core.field import VectorField, ScalarField
from physics.navier_stokes.terms import (
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
    AccelerationTerm,
)


class NavierStokesSolver:
    """Navier-Stokes方程式のシンプルな解法

    入力された速度場、密度場、粘性場、圧力場から速度の時間微分を計算する。
    """

    def __init__(
        self,
        enable_surface_tension: bool = True,
        enable_gravity: bool = True,
    ):
        """
        ソルバーを初期化

        Args:
            enable_surface_tension: 表面張力の有効化
            enable_gravity: 重力の有効化
        """
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
        external_force: VectorField,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        速度の時間微分を計算

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            pressure: 圧力場
            levelset: レベルセット関数（オプション）
            **kwargs: 追加のパラメータ

        Returns:
            各方向の速度成分の時間微分
        """
        # 各項の計算
        advection = self.advection_term.compute(velocity)
        diffusion = self.diffusion_term.compute(velocity)
        pressure_grad = self.pressure_term.compute(velocity, pressure)

        # 速度の時間微分を統合
        velocity_derivative = [
            -adv + diff - press_grad + f
            for adv, diff, press_grad, f in zip(
                advection,
                diffusion,
                pressure_grad,
                external_force
            )
        ]

        return velocity_derivative

    def compute_timestep(
        self, velocity: VectorField, density: ScalarField, pressure: ScalarField
    ) -> float:
        """
        安定な時間刻み幅を計算

        Args:
            velocity: 速度場
            density: 密度場
            pressure: 圧力場

        Returns:
            最大許容時間刻み幅
        """
        # 各項からの時間刻み幅の制限を取得
        timesteps = [
            self.advection_term.compute_timestep(velocity),
            self.diffusion_term.compute_timestep(velocity),
            self.pressure_term.compute_timestep(velocity, pressure),
        ]

        # 最も厳しい制限を採用
        return min(timesteps)

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        診断情報を取得

        Returns:
            ソルバーの診断情報
        """
        # 各項の診断情報を収集
        return {
            "advection": self.advection_term.get_diagnostics(),
            "diffusion": self.diffusion_term.get_diagnostics(),
            "pressure": self.pressure_term.get_diagnostics(),
            "force_terms": [term.get_diagnostics() for term in self.force_terms],
        }
