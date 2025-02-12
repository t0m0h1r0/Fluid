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

    def __init__(
        self, 
        name: str = "Pressure", 
        enabled: bool = True,
        order: int = 2
    ):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
            order: 差分近似の次数（2次、4次など）
        """
        super().__init__(name, enabled)
        self._order = order
        self._diagnostics: Dict[str, Any] = {}

    def compute(
        self, 
        velocity: VectorField, 
        pressure: ScalarField, 
        density: Union[float, ScalarField, None] = None,
        **kwargs
    ) -> VectorField:
        """
        圧力項の寄与を計算

        Args:
            velocity: 速度場
            pressure: 圧力場
            density: 密度（定数、スカラー場、またはNone）

        Returns:
            圧力項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 結果用のVectorFieldを作成
        result = VectorField(velocity.shape, velocity.dx)
        dx = velocity.dx

        # 密度の設定（デフォルトは1000.0）
        rho = 1000.0 if density is None else density

        # 密度の処理（スカラー場または定数）
        if isinstance(rho, ScalarField):
            rho_data = rho.data
        else:
            rho_data = rho

        # 各方向の圧力勾配項を計算
        pressure_gradient_terms = []
        for i in range(velocity.ndim):
            # 圧力勾配の計算: -1/ρ ∇p
            grad_p = np.gradient(pressure.data, dx, axis=i)
            
            # 密度による除算
            pressure_grad = (
                -grad_p / np.maximum(rho_data, 1e-10)  # ゼロ除算を防ぐ
            )
            
            # 結果をVectorFieldに設定
            result.components[i].data = pressure_grad
            pressure_gradient_terms.append(pressure_grad)

        # 診断情報の更新
        self._update_diagnostics(result, pressure_gradient_terms, pressure)

        return result

    def _update_diagnostics(
        self, 
        result: VectorField, 
        pressure_gradient_terms: list,
        pressure: ScalarField
    ):
        """
        診断情報を更新

        Args:
            result: 計算された圧力項
            pressure_gradient_terms: 各成分の圧力勾配項
            pressure: 圧力場
        """
        max_gradient = [np.max(np.abs(term)) for term in pressure_gradient_terms]
        self._diagnostics = {
            "order": self._order,
            "max_gradient_x": float(max_gradient[0]) if len(max_gradient) > 0 else 0.0,
            "max_gradient_y": float(max_gradient[1]) if len(max_gradient) > 1 else 0.0,
            "max_gradient_z": float(max_gradient[2]) if len(max_gradient) > 2 else 0.0,
            "max_pressure_gradient": float(
                max(np.max(np.abs(comp.data)) for comp in result.components)
            ),
            "pressure_range": {
                "min": float(np.min(pressure.data)),
                "max": float(np.max(pressure.data)),
                "mean": float(np.mean(pressure.data)),
            }
        }

    def compute_timestep(self, velocity: VectorField, pressure: ScalarField, **kwargs) -> float:
        """
        圧力項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場
            pressure: 圧力場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 音速の推定
        # 音速 c = √(dp/dρ) ≈ √(max(p)/ρ)
        density = kwargs.get('density', 1000.0)
        max_pressure = float(np.max(pressure.data))
        sound_speed = np.sqrt(max_pressure / density)

        # CFL条件に基づく時間刻み幅の計算
        cfl = kwargs.get("cfl", 0.5)
        return cfl * velocity.dx / (sound_speed + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag