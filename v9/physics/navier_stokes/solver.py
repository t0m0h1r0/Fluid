from typing import Dict, Any, Optional
import numpy as np

from core.field import VectorField, ScalarField
from physics.navier_stokes.terms import (
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
)


class NavierStokesSolver:
    """Navier-Stokes方程式のシンプルな解法

    入力された速度場、密度場、粘性場、圧力場から速度の時間微分を計算する。
    すべての入出力をVectorField形式で扱い、一貫性のある処理を提供する。
    """

    def __init__(self):
        """ソルバーを初期化"""
        # 各項の初期化
        self.advection_term = AdvectionTerm()
        self.diffusion_term = DiffusionTerm()
        self.pressure_term = PressureTerm()

    def compute_velocity_derivative(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        pressure: ScalarField,
        force: Optional[VectorField] = None,
        **kwargs,
    ) -> VectorField:
        """速度の時間微分を計算

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

        # 各項の計算（すべてVectorField型で返される）
        advection = self.advection_term.compute(velocity)
        diffusion = self.diffusion_term.compute(velocity)
        pressure_grad = self.pressure_term.compute(velocity, pressure)

        # 外力項の処理
        if force is None:
            force = VectorField(velocity.shape, velocity.dx)

        # 各成分にを統合、速度場の時間微分を計算
        for i in range(velocity.ndim):
            result.components[i].data = (
                -advection.components[i].data
                + diffusion.components[i].data
                - pressure_grad.components[i].data
                + force.components[i].data
            )

        # 診断情報の更新
        self._update_diagnostics(
            advection=advection,
            diffusion=diffusion,
            pressure_grad=pressure_grad,
            force=force,
            result=result,
        )

        return result

    def _update_diagnostics(self, **fields: VectorField) -> None:
        """診断情報を更新

        Args:
            **fields: 更新に使用する各VectorField
        """
        for name, field in fields.items():
            max_value = max(np.max(np.abs(comp.data)) for comp in field.components)
            self._diagnostics = (
                self._diagnostics if hasattr(self, "_diagnostics") else {}
            )
            self._diagnostics[f"{name}_max"] = float(max_value)

    def compute_timestep(
        self, velocity: VectorField, density: ScalarField, pressure: ScalarField
    ) -> float:
        """安定な時間刻み幅を計算

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
        dt = min(timesteps)

        # 診断情報に時間刻み幅の情報を追加
        if not hasattr(self, "_diagnostics"):
            self._diagnostics = {}
        self._diagnostics["timestep"] = dt

        return dt

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得

        Returns:
            ソルバーの診断情報
        """
        # 各項の診断情報を収集
        diagnostics = {
            "advection": self.advection_term.get_diagnostics(),
            "diffusion": self.diffusion_term.get_diagnostics(),
            "pressure": self.pressure_term.get_diagnostics(),
        }

        # ソルバー全体の診断情報を追加
        if hasattr(self, "_diagnostics"):
            diagnostics["solver"] = self._diagnostics

        return diagnostics
