"""
圧力項の計算を提供するモジュール

Navier-Stokes方程式における -1/ρ ∇p 項を計算します。
"""

from typing import Dict, Any, Union
import numpy as np

from core.field import VectorField, ScalarField
from .base import BaseNavierStokesTerm


class PressureTerm(BaseNavierStokesTerm):
    """
    圧力項を計算するクラス

    速度場の圧力勾配項を中心差分で近似計算します。
    """

    def __init__(self, name: str = "Pressure", enabled: bool = True, order: int = 2):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
            order: 差分近似の次数（2次、4次など）
        """
        super().__init__(name, enabled)
        self._order = order

    def compute(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        density: Union[float, ScalarField, None] = None,
        **kwargs,
    ) -> VectorField:
        """
        圧力項 -1/ρ ∇p を計算

        Args:
            velocity: 速度場（形状情報のみ使用）
            pressure: 圧力場
            density: 密度場（またはスカラー値）

        Returns:
            圧力項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 密度場の正規化
        density_field = (
            density
            if isinstance(density, ScalarField)
            else ScalarField(velocity.shape, velocity.dx, initial_value=density or 1000.0)
        )

        # 圧力勾配項の計算
        result = -pressure.gradient() / density_field

        # 診断情報の更新
        self._update_diagnostics(result, pressure, density_field)

        return result

    def _update_diagnostics(
        self, result: VectorField, pressure: ScalarField, density: ScalarField
    ):
        """
        診断情報を更新

        Args:
            result: 計算された圧力項
            pressure: 圧力場
            density: 密度場
        """
        gradient_max = [float(np.max(np.abs(comp.data))) for comp in result.components]

        self._diagnostics = {
            "order": self._order,
            "max_gradient_x": gradient_max[0] if len(gradient_max) > 0 else 0.0,
            "max_gradient_y": gradient_max[1] if len(gradient_max) > 1 else 0.0,
            "max_gradient_z": gradient_max[2] if len(gradient_max) > 2 else 0.0,
            "max_pressure_gradient": float(max(gradient_max)),
            "pressure_range": {
                "min": float(np.min(pressure.data)),
                "max": float(np.max(pressure.data)),
                "mean": float(np.mean(pressure.data)),
            },
            "density_range": {
                "min": float(np.min(density.data)),
                "max": float(np.max(density.data)),
            },
        }

    def compute_timestep(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        density: Union[float, ScalarField, None] = None,
        **kwargs,
    ) -> float:
        """
        圧力項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場
            pressure: 圧力場
            density: 密度場またはスカラー値

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 密度の処理
        if density is None:
            rho = 1000.0
        elif isinstance(density, ScalarField):
            rho = np.min(density.data)  # 安全側の評価
        else:
            rho = float(density)

        # 音速の推定: c ≈ √(max(|∇p|)/ρ)
        max_pressure_grad = max(
            np.max(np.abs(pressure.gradient(i))) for i in range(pressure.ndim)
        )
        sound_speed = np.sqrt(max_pressure_grad / (rho + 1e-10))

        # CFL条件に基づく時間刻み幅の計算
        cfl = kwargs.get("cfl", 0.5)
        return cfl * velocity.dx / (sound_speed + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag