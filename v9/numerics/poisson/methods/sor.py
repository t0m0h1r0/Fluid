"""
逐次過緩和法 (SOR: Successive Over-Relaxation) による
Poisson方程式の数値解法

理論的背景:
- ガウス・ザイデル法の収束を加速する反復法
- 緩和パラメータ(ω)により収束特性を制御
- 対称正定値行列に対して効果的

数学的定式化:
1. ガウス・ザイデル法の更新則を拡張
2. 緩和パラメータによる加速
3. 収束条件の厳密な評価

主な特徴:
- パラメータチューニングによる高速収束
- メモリ効率が良い
- 並列化が比較的容易
"""

from typing import Optional, Dict, Any

import jax.numpy as jnp
from jax import jit
from functools import partial

from core.field import ScalarField
from .base import PoissonSolverBase, PoissonSolverConfig


class PoissonSORSolver(PoissonSolverBase):
    """
    SOR法によるPoissonソルバー

    逐次過緩和法を用いたPoisson方程式の数値解法
    """

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        relaxation_parameter: float = 1.5,
    ):
        """
        SORソルバーを初期化

        Args:
            config: ソルバー設定
            relaxation_parameter: 緩和パラメータ (0 < ω < 2)
                - ω = 1: ガウス・ザイデル法
                - 1 < ω < 2: 加速された反復法
                - 最適値は問題依存
        """
        super().__init__(config or PoissonSolverConfig())

        # 緩和パラメータの検証
        if not 0 < relaxation_parameter <= 2:
            raise ValueError("緩和パラメータは0から2の間である必要があります")

        self.relaxation_parameter = relaxation_parameter

    @partial(jit, static_argnums=(0,))
    def _compute_laplacian(self, field: ScalarField) -> ScalarField:
        """
        ラプラシアン演算子を計算

        Args:
            field: 入力スカラー場

        Returns:
            ラプラシアンを適用した結果
        """
        result = ScalarField(field.shape, field.dx)

        for axis in range(field.ndim):
            # 中心差分による2階微分
            grad1 = jnp.gradient(field.data, field.dx[axis], axis=axis)
            grad2 = jnp.gradient(grad1, field.dx[axis], axis=axis)
            result.data += grad2

        return result

    @partial(jit, static_argnums=(0,))
    def solve(
        self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None
    ) -> ScalarField:
        """
        SOR法によりPoisson方程式を解く

        Args:
            rhs: 右辺項 f
            initial_guess: 初期推定解（省略時はゼロベクトル）

        Returns:
            解 u
        """
        # 入力の妥当性検証
        self.validate_input(rhs, initial_guess)

        # 初期解の準備
        solution = (
            initial_guess.copy()
            if initial_guess is not None
            else ScalarField(rhs.shape, rhs.dx, initial_value=0.0)
        )

        # SOR法の反復関数
        def sor_step(solution_data):
            """
            単一のSORステップを実行

            Args:
                solution_data: 現在の解のデータ

            Returns:
                更新された解のデータ
            """
            # 各次元に対して更新
            for axis in range(solution.ndim):
                # シフトによる近傍点の取得
                forward_shift = jnp.roll(solution_data, -1, axis=axis)
                backward_shift = jnp.roll(solution_data, 1, axis=axis)

                # 対角項以外の項の和
                neighbors_sum = forward_shift + backward_shift

                # SOR更新則
                # u_new = (1-ω)u_old + ω * (f - Σ(u_j)) / (2*ndim)
                solution_data = (
                    1 - self.relaxation_parameter
                ) * solution_data + self.relaxation_parameter * (
                    rhs.data
                    - (
                        self._compute_laplacian(
                            ScalarField(solution.shape, solution.dx, solution_data)
                        ).data
                    )
                ) / (2 * solution.ndim)

            return solution_data

        # メイン反復ループ
        for _ in range(self.config.max_iterations):
            # SORステップの実行
            old_solution = solution.data.copy()
            solution.data = sor_step(solution.data)

            # 収束判定
            residual = self._compute_laplacian(solution)
            residual.data -= rhs.data

            # 残差のノルムを計算
            residual_norm = jnp.linalg.norm(residual.data)
            rhs_norm = jnp.linalg.norm(rhs.data)

            # 収束判定
            if residual_norm <= self.config.tolerance * rhs_norm:
                break

        return solution

    def compute_residual(self, solution: ScalarField, rhs: ScalarField) -> float:
        """
        残差のノルムを計算

        Args:
            solution: 解
            rhs: 右辺項

        Returns:
            残差のノルム
        """
        # ラプラシアンと残差の計算
        residual = self._compute_laplacian(solution)
        residual.data -= rhs.data

        return float(jnp.linalg.norm(residual.data))

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        ソルバーの診断情報を取得

        Returns:
            診断情報の辞書
        """
        # 親クラスの診断情報を取得し拡張
        diag = super().get_diagnostics()
        diag.update(
            {
                "relaxation_parameter": self.relaxation_parameter,
                "method": "Successive Over-Relaxation (SOR)",
            }
        )
        return diag
