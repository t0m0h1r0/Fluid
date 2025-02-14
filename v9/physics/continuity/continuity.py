"""
連続の方程式の数値実装

支配方程式:
保存形: ∂f/∂t + ∇⋅(uf) = 0
非保存形: ∂f/∂t + u⋅∇f = 0

このモジュールは以下の数値的課題に対応:
1. スカラー場の移流
2. 保存形と非保存形の計算
3. 移流の安定性評価
"""

from typing import Dict, Any
from jax import jit

from core.field import ScalarField, VectorField, GridInfo


class ContinuityEquation:
    """
    連続の方程式の数値解法

    主な機能:
    - スカラー場の移流計算
    - 保存形・非保存形の選択
    - 数値安定性の評価
    """

    def __init__(self, conservative: bool = False, epsilon: float = 1e-10):
        """
        連続の方程式ソルバーを初期化

        Args:
            conservative: 保存形で計算するかどうか
            epsilon: 数値安定化のための小さな値
        """
        self.conservative = conservative
        self._epsilon = epsilon

    @jit
    def compute_derivative(
        self,
        field: ScalarField,
        velocity: VectorField,
    ) -> ScalarField:
        """
        スカラー場の時間微分を計算

        Args:
            field: 移流される任意のスカラー場
            velocity: 速度場

        Returns:
            スカラー場の時間微分

        数式:
        - 保存形: ∂f/∂t = -∇⋅(uf)
        - 非保存形: ∂f/∂t = -u⋅∇f
        """
        return (
            self._compute_conservative(field, velocity)
            if self.conservative
            else self._compute_nonconservative(field, velocity)
        )

    @jit
    def _compute_conservative(
        self,
        field: ScalarField,
        velocity: VectorField,
    ) -> ScalarField:
        """
        保存形の移流項を計算: -∇⋅(uf)

        Args:
            field: スカラー場
            velocity: 速度場

        Returns:
            保存形の時間微分
        """
        # フラックスの計算: uf
        flux = velocity * field

        # 発散の計算: ∇⋅(uf)
        divergence = flux.divergence()

        # 時間微分: ∂f/∂t = -∇⋅(uf)
        return -divergence

    @jit
    def _compute_nonconservative(
        self,
        field: ScalarField,
        velocity: VectorField,
    ) -> ScalarField:
        """
        非保存形の移流項を計算: -u⋅∇f

        Args:
            field: スカラー場
            velocity: 速度場

        Returns:
            非保存形の時間微分
        """
        # 移流項の計算: u⋅∇f
        advection = velocity * field.gradient()

        # 時間微分: ∂f/∂t = -u⋅∇f
        return -advection

    def compute_cfl_timestep(
        self, velocity: VectorField, grid_info: GridInfo, cfl: float = 0.5
    ) -> float:
        """
        CFL条件に基づく時間刻み幅を計算

        数値的安定性条件の評価:
        Δt ≤ CFL * (Δx / |u|_max)

        Args:
            velocity: 速度場
            grid_info: グリッド情報
            cfl: CFL数（デフォルト: 0.5）

        Returns:
            安定な時間刻み幅
        """
        # 最大速度の計算
        max_velocity = velocity.magnitude().max()

        if max_velocity > 0:
            # CFL条件に基づく時間刻み幅
            min_dx = min(grid_info.dx)
            return cfl * min_dx / (max_velocity + self._epsilon)
        else:
            return float("inf")

    def get_diagnostics(
        self,
        field: ScalarField,
        velocity: VectorField,
        derivative: ScalarField,
    ) -> Dict[str, Any]:
        """
        移流計算の診断情報を取得

        Args:
            field: スカラー場
            velocity: 速度場
            derivative: 計算された時間微分

        Returns:
            診断情報の辞書
        """
        return {
            "method": "conservative" if self.conservative else "nonconservative",
            "field_stats": {
                "min": field.min(),
                "max": field.max(),
                "mean": field.mean(),
            },
            "velocity_stats": {
                "max_magnitude": velocity.magnitude().max(),
                "mean_magnitude": velocity.magnitude().mean(),
            },
            "derivative_stats": {
                "max": derivative.max(),
                "min": derivative.min(),
                "mean": derivative.mean(),
            },
            "numerical_properties": {
                "epsilon": self._epsilon,
                "cfl_method": "CFL条件による時間刻み幅制限",
            },
        }
