"""
二相流のナビエ・ストークス方程式における速度場の加速度項

支配方程式:
∂(ρu)/∂t + ∇⋅(ρu⊗u) = -∇p + ∇⋅τ + ρg + Fσ

速度場の時間発展に必要な項の統合を行う
"""

from typing import List, Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from .base import BaseNavierStokesTerm
from ..terms.advection import AdvectionTerm
from ..terms.diffusion import DiffusionTerm
from ..terms.pressure import PressureTerm
from ..terms.force import GravityForce


class AccelerationTerm(BaseNavierStokesTerm):
    """
    速度場の加速度項を計算するクラス

    Navier-Stokes方程式の右辺を統合して速度場の時間微分を計算
    """

    def __init__(
        self, 
        name: str = "Acceleration", 
        enabled: bool = True
    ):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)
        
        # 各項のインスタンスを事前に作成
        self._advection_term = AdvectionTerm()
        self._pressure_term = PressureTerm()
        self._diffusion_term = DiffusionTerm()
        self._gravity_term = GravityForce()

    def compute(
        self,
        velocity: VectorField, 
        density: ScalarField,
        viscosity: ScalarField,
        pressure: ScalarField,
        **kwargs
    ) -> List[np.ndarray]:
        """
        速度場の時間微分（加速度）を計算

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            pressure: 圧力場
            **kwargs: 追加のパラメータ

        Returns:
            各方向の速度の時間微分（加速度）
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 各項を計算
        advection = self._advection_term.compute(velocity)
        pressure_term = self._pressure_term.compute(velocity, pressure)
        diffusion = self._diffusion_term.compute(velocity)
        gravity = self._gravity_term.compute(velocity, density)

        # 項を統合
        acceleration = [
            -adv + press + diff + grav
            for adv, press, diff, grav in zip(
                advection, pressure_term, diffusion, gravity
            )
        ]

        # 診断情報の更新
        self._diagnostics = {
            'advection': self._advection_term.get_diagnostics(),
            'pressure': self._pressure_term.get_diagnostics(),
            'diffusion': self._diffusion_term.get_diagnostics(),
            'gravity': self._gravity_term.get_diagnostics(),
            'max_acceleration': float(max(np.max(np.abs(a)) for a in acceleration))
        }

        return acceleration

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        診断情報を取得

        Returns:
            加速度項の診断情報
        """
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag