"""
移流項（対流項）の計算を提供するモジュール

Navier-Stokes方程式における u⋅∇u 項を計算します。
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField
from .base import BaseNavierStokesTerm


class AdvectionTerm(BaseNavierStokesTerm):
    """
    移流項（対流項）を計算するクラス

    速度場の移流（u⋅∇u）を中心差分で近似計算します。
    """

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
            scheme: 差分スキーム（現時点では中心差分のみ）
        """
        super().__init__(name, enabled)
        self._scheme = scheme

    def compute(self, velocity: VectorField, **kwargs) -> VectorField:
        """
        移流項の寄与を計算 (u⋅∇)u

        Args:
            velocity: 速度場

        Returns:
            移流項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        result = VectorField(velocity.shape, velocity.dx)

        # (u⋅∇)u の計算
        for i, v_i in enumerate(velocity.components):
            # u_j ∂u_i/∂x_j を計算
            advection = np.zeros_like(v_i.data)
            for j, v_j in enumerate(velocity.components):
                advection += v_j.data * v_i.gradient(j)

            result.components[i].data = -advection  # 符号に注意

        # 診断情報の更新
        self._update_diagnostics(result)

        return result

    def _update_diagnostics(self, result: VectorField):
        """
        診断情報を更新

        Args:
            result: 計算された移流項
        """
        self._diagnostics = {
            "scheme": self._scheme,
            "max_advection": float(
                max(np.max(np.abs(comp.data)) for comp in result.components)
            ),
            "component_max": {
                f"component_{i}": float(np.max(np.abs(comp.data)))
                for i, comp in enumerate(result.components)
            },
        }

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """
        移流項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 最大速度の計算（各成分の最大絶対値）
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # CFLに基づく時間刻み幅の計算
        cfl = kwargs.get("cfl", 0.5)
        return cfl * velocity.dx / (max_velocity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag
