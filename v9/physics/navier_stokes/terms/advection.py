"""
移流項（対流項）の計算を提供するモジュール

Navier-Stokes方程式における移流項 (u·∇)u の計算を実装します。
ScalarField/VectorFieldの演算子を活用して、
数学的な表現に近い形で実装を提供します。

数学的表現:
(u·∇)u_i = Σ_j (u_j ∂u_i/∂x_j)

ここで:
- u_j: j方向の速度成分
- ∂u_i/∂x_j: i成分のj方向への偏微分
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField
from .base import BaseNavierStokesTerm


class AdvectionTerm(BaseNavierStokesTerm):
    """移流項を計算するクラス"""

    def __init__(
        self,
        name: str = "Advection",
        enabled: bool = True,
        scheme: str = "central",
    ):
        """移流項計算器を初期化

        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
            scheme: 差分スキーム
        """
        super().__init__(name, enabled)
        self._scheme = scheme

    def compute(self, velocity: VectorField, **kwargs) -> VectorField:
        """移流項 -(u·∇)u を計算

        Args:
            velocity: 速度場
            **kwargs: 追加のパラメータ

        Returns:
            計算された移流項（VectorField）
        """
        if not self.enabled:
            return VectorField(velocity.shape[:-1], velocity.dx)

        # 計算結果を格納するベクトル場
        result = VectorField(velocity.shape[:-1], velocity.dx)

        # 速度成分の数を取得
        n_components = len(velocity.shape[:-1])  # VectorFieldの形状から最後の次元を除外

        # 各速度成分について -(u·∇)u_i を計算
        for i in range(n_components):
            # 各方向の寄与を計算して合計
            advection = np.zeros(velocity.shape[:-1])
            for j in range(n_components):
                # u_j * ∂u_i/∂x_j を計算
                # データに直接アクセスして高速化
                velocity_j = velocity._data[..., j]
                grad_u_i = np.gradient(velocity._data[..., i], velocity.dx[j], axis=j)
                advection += velocity_j * grad_u_i

            # 結果を設定（負符号に注意）
            result._data[
                ..., i
            ] = -advection  # 移流項は運動方程式の左辺に移項されるため負符号

        # 診断情報の更新
        self._update_diagnostics(result)

        return result

    def _update_diagnostics(self, result: VectorField):
        """診断情報を更新

        Args:
            result: 計算された移流項
        """
        self._diagnostics = {
            "scheme": self._scheme,
            "max_magnitude": float(result.magnitude().max()),
            "component_max": {
                f"component_{i}": float(np.max(np.abs(result._data[..., i])))
                for i in range(result.shape[-1])
            },
        }

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """移流による時間刻み幅の制限を計算

        CFL条件に基づいて、数値的に安定な時間刻み幅を計算します。
        Δt ≤ CFL * Δx / |u|_max

        Args:
            velocity: 速度場
            **kwargs: 追加のパラメータ（cflなど）

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 最大速度を計算
        max_velocity = velocity.magnitude().max()

        # CFL条件に基づく時間刻み幅の計算
        cfl = kwargs.get("cfl", 0.5)
        return cfl * min(velocity.dx) / (max_velocity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag
