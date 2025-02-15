"""
連続の方程式の数値実装

Level Set法と多相流体シミュレーションのための連続の方程式モジュール
"""

from typing import Dict, Any
import jax
import jax.numpy as jnp

from core.field import ScalarField, VectorField, GridInfo, FieldFactory


class ContinuityEquation:
    """
    連続の方程式のための数値的解法クラス

    Level Set法における連続の方程式（保存形・非保存形）の数値計算を担当
    """

    def __init__(self, conservative: bool = False, epsilon: float = 1e-10):
        """
        連続の方程式ソルバーを初期化

        Args:
            conservative: 保存形で計算するかどうか
            epsilon: 数値安定化のための小さな値
        """
        self.conservative = conservative
        self.epsilon = epsilon

    @staticmethod
    # @jax.jit
    def _compute_conservative(
        scalar_data: jnp.ndarray, velocity_data: jnp.ndarray, dx: jnp.ndarray
    ) -> jnp.ndarray:
        """
        保存形の移流項を計算（低レベルの純粋関数）

        Args:
            scalar_data: スカラー場データ
            velocity_data: 速度ベクトル場データ
            dx: グリッド間隔

        Returns:
            保存形の時間微分
        """
        # 各方向の移流フラックスを計算
        flux = velocity_data * scalar_data

        # 発散の計算
        divergence = jnp.zeros_like(scalar_data)
        for i in range(scalar_data.ndim):
            # 中心差分による勾配計算
            divergence += jnp.gradient(flux[i], dx[i], axis=i)

        return -divergence

    @staticmethod
    # @jax.jit
    def _compute_nonconservative(
        scalar_data: jnp.ndarray, velocity_data: jnp.ndarray, dx: jnp.ndarray
    ) -> jnp.ndarray:
        """
        非保存形の移流項を計算（低レベルの純粋関数）

        Args:
            scalar_data: スカラー場データ
            velocity_data: 速度ベクトル場データ
            dx: グリッド間隔

        Returns:
            非保存形の時間微分
        """
        # スカラー場の勾配を計算
        grad = jnp.stack(
            [jnp.gradient(scalar_data, dx[i], axis=i) for i in range(scalar_data.ndim)]
        )

        # u⋅∇f の計算
        advection = jnp.sum(velocity_data * grad, axis=0)

        return -advection

    # @jax.jit
    def compute_derivative(
        self, field: ScalarField, velocity: VectorField
    ) -> ScalarField:
        """
        スカラー場の時間微分を計算

        Args:
            field: 移流されるスカラー場
            velocity: 速度場

        Returns:
            スカラー場の時間微分
        """
        # 入力データのJAX配列への変換
        scalar_data = jnp.asarray(field.data)
        velocity_data = jnp.stack(
            [jnp.asarray(comp.data) for comp in velocity.components]
        )

        # グリッド間隔の準備
        dx = jnp.array(field.dx)

        # 計算モードに応じた関数呼び出し
        result_data = jax.lax.cond(
            self.conservative,
            lambda _: self._compute_conservative(scalar_data, velocity_data, dx),
            lambda _: self._compute_nonconservative(scalar_data, velocity_data, dx),
            None,
        )

        # グリッド情報を使用して新しいScalarFieldを作成
        result_grid = GridInfo(
            shape=field.shape,
            dx=field.dx,
            time=field.grid.time + 1,  # タイムステップを進める
        )
        return FieldFactory.create_scalar_field(result_grid, result_data)

    def compute_cfl_timestep(self, velocity: VectorField, cfl: float = 0.5) -> float:
        """
        CFL条件に基づく時間刻み幅を計算

        Args:
            velocity: 速度場
            cfl: CFL数

        Returns:
            安定な時間刻み幅
        """
        # 速度の大きさを計算
        magnitude = jnp.sqrt(
            sum(jnp.asarray(comp.data) ** 2 for comp in velocity.components)
        )
        max_velocity = float(jnp.max(magnitude))

        # 最小グリッド間隔の計算
        min_dx = float(min(velocity.dx))

        # 安定な時間刻み幅を返却
        return (
            min_dx / (max_velocity + self.epsilon) * cfl
            if max_velocity > 0
            else float("inf")
        )

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
        }
