"""
圧力勾配項の計算を提供するモジュール

流体の運動方程式における圧力勾配項 -1/ρ ∇p を計算します。
"""

from typing import Optional, Dict, Any, Union
import numpy as np

from core.field import VectorField, ScalarField
from .base import PoissonTerm


class PressureTerm(PoissonTerm):
    """圧力勾配項の計算クラス

    流体の運動方程式における圧力勾配項 -1/ρ ∇p を計算します。
    これは流体に作用する圧力による力を表します。
    """

    def __init__(self, name: str = "PressureGradient", enabled: bool = True):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)

    def compute(
        self,
        pressure: ScalarField,
        density: Optional[Union[float, ScalarField]] = None,
        **kwargs,
    ) -> VectorField:
        """圧力勾配項 -1/ρ ∇p を計算

        Args:
            pressure: 圧力場
            density: 密度場（スカラー値またはScalarField）
            **kwargs: 追加のパラメータ

        Returns:
            圧力勾配項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(pressure.shape, pressure.dx)

        # 密度場の準備
        if density is None:
            density = ScalarField(pressure.shape, pressure.dx, initial_value=1.0)
        elif isinstance(density, (int, float)):
            density = ScalarField(
                pressure.shape, pressure.dx, initial_value=float(density)
            )

        # 圧力勾配の計算
        result = VectorField(pressure.shape, pressure.dx)
        for i in range(pressure.ndim):
            # i方向の圧力勾配を計算: ∂p/∂x_i
            grad_p = np.gradient(pressure.data, pressure.dx[i], axis=i)
            # -1/ρ ∂p/∂x_i を設定
            result.components[i].data = -grad_p / density.data

        # 診断情報の更新
        self._update_diagnostics(result, pressure, density)

        return result

    def _update_diagnostics(
        self, result: VectorField, pressure: ScalarField, density: ScalarField
    ):
        """診断情報を更新

        Args:
            result: 計算された圧力勾配項
            pressure: 圧力場
            density: 密度場
        """
        self._diagnostics = {
            "pressure_range": {
                "min": float(np.min(pressure.data)),
                "max": float(np.max(pressure.data)),
                "mean": float(np.mean(pressure.data)),
            },
            "density_range": {
                "min": float(np.min(density.data)),
                "max": float(np.max(density.data)),
            },
            "gradient_max": {
                f"component_{i}": float(np.max(np.abs(comp.data)))
                for i, comp in enumerate(result.components)
            },
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag
