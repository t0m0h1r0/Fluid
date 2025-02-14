"""
移流項（対流項）の計算を提供するモジュール（改良版）

Navier-Stokes方程式における u⋅∇u 項を計算します。
新しい演算子とメソッドを活用して実装を改善しています。
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField
from .base import BaseNavierStokesTerm


class AdvectionTerm(BaseNavierStokesTerm):
    """移流項（対流項）を計算するクラス（改良版）"""

    def __init__(
        self,
        name: str = "Advection",
        enabled: bool = True,
        scheme: str = "central",
    ):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
            scheme: 差分スキーム
        """
        super().__init__(name, enabled)
        self._scheme = scheme

    def compute(self, velocity: VectorField, **kwargs) -> VectorField:
        """
        移流項の寄与を計算 (u⋅∇)u
        新しい演算子 @ を使用して内積を計算

        Args:
            velocity: 速度場

        Returns:
            移流項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # @演算子を使用した内積計算
        result = -(velocity @ velocity.gradient())

        # 診断情報の更新
        self._update_diagnostics(result)
        return result

    def _update_diagnostics(self, result: VectorField):
        """診断情報を更新（新しいメソッドを活用）"""
        self._diagnostics = {
            "scheme": self._scheme,
            "max_magnitude": float(result.magnitude().max()),
            "component_max": {
                f"component_{i}": float(comp.norm(ord=np.inf))
                for i, comp in enumerate(result.components)
            },
        }

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """
        移流項に基づく時間刻み幅の制限を計算
        新しいメソッドを活用して最大速度を計算

        Args:
            velocity: 速度場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # magnitude()メソッドを使用して最大速度を計算
        max_velocity = velocity.magnitude().max()

        # CFLに基づく時間刻み幅の計算
        cfl = kwargs.get("cfl", 0.5)
        return cfl * min(velocity.dx) / (max_velocity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag
