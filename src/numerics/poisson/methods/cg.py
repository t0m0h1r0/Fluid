"""
共役勾配法によるPoisson方程式の数値解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union
import jax
import jax.numpy as jnp
import jax.lax as lax

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField


class PoissonCGSolver(PoissonSolverBase):
    """共役勾配法によるPoissonソルバー（JAX最適化版）"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
    ):
        """
        Args:
            config: ソルバーの設定
        """
        # デフォルト設定の調整
        if config is None:
            config = PoissonSolverConfig(
                max_iterations=500,  # デフォルトの反復回数を増加
                tolerance=1e-8,  # より厳しい収束判定
            )
        super().__init__(config)

        # JAXの最適化設定
        self._jax_config()

    def _jax_config(self):
        """JAXの最適化設定"""
        # 64ビット浮動小数点数を使用（高精度計算）
        jax.config.update("jax_enable_x64", True)

    def solve(
        self,
        rhs: Union[jnp.ndarray, ScalarField],
        initial_guess: Optional[Union[jnp.ndarray, ScalarField]] = None,
    ) -> jnp.ndarray:
        """Poisson方程式を解く

        Args:
            rhs: 右辺項
            initial_guess: 初期推定解（オプション）

        Returns:
            解
        """
        # 入力の検証と配列への変換
        rhs_array, initial_array = self.validate_input(rhs, initial_guess)

        # 初期解の準備
        solution = (
            initial_array if initial_array is not None else jnp.zeros_like(rhs_array)
        )

        # グリッド間隔
        dx = jnp.array(self.config.dx)
        dx2 = jnp.square(dx)

        # グリッド間隔をキャプチャしたラプラシアン演算子
        @jax.jit
        def laplacian_operator(u: jnp.ndarray) -> jnp.ndarray:
            """ラプラシアン演算子のJIT最適化版"""

            # 6次精度の中心差分スキーム
            def central_diff_6th_order(arr, axis, dx2_axis):
                """6次精度の中心差分スキーム"""
                diff = (
                    -1 / 60 * jnp.roll(arr, 3, axis=axis)
                    + 3 / 20 * jnp.roll(arr, 2, axis=axis)
                    - 3 / 4 * jnp.roll(arr, 1, axis=axis)
                    + 3 / 4 * jnp.roll(arr, -1, axis=axis)
                    - 3 / 20 * jnp.roll(arr, -2, axis=axis)
                    + 1 / 60 * jnp.roll(arr, -3, axis=axis)
                )
                return diff / dx2_axis

            # 各方向のラプラシアンを計算
            return (
                central_diff_6th_order(u, axis=0, dx2_axis=dx2[0])
                + central_diff_6th_order(u, axis=1, dx2_axis=dx2[1])
                + central_diff_6th_order(u, axis=2, dx2_axis=dx2[2])
            )

        # 前処理関数
        @jax.jit
        def preconditioner(r: jnp.ndarray) -> jnp.ndarray:
            """対角前処理の計算"""
            # ラプラシアン演算子の対角成分を近似
            diag_approx = (2 / dx2[0] + 2 / dx2[1] + 2 / dx2[2]) * jnp.ones_like(r)

            return r / (diag_approx + 1e-10)

        # 内積計算関数
        @jax.jit
        def dot_product(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            """ベクトルの内積を計算"""
            return jnp.sum(x * y)

        # CG法の反復関数（前処理付き）
        @jax.jit
        def cg_iteration(carry):
            """単一のCG反復（前処理付き共役勾配法）"""
            solution, residual, direction, rz_old, rhs = carry

            # 前処理付きの残差
            z = preconditioner(residual)

            # 内積の計算
            rz = dot_product(residual, z)

            # A * p の計算
            Ap = laplacian_operator(direction)

            # ステップサイズの計算
            dAd = dot_product(direction, Ap)
            alpha = rz / (dAd + 1e-15)

            # 解と残差の更新
            new_solution = solution + alpha * direction
            new_residual = residual - alpha * Ap

            # 新しい内積の計算
            z_new = preconditioner(new_residual)
            rz_new = dot_product(new_residual, z_new)

            # 方向ベクトルの更新
            beta = rz_new / (rz + 1e-15)
            new_direction = z_new + beta * direction

            # 残差ノルムの計算
            residual_norm = jnp.sqrt(rz_new)

            return (
                new_solution,
                new_residual,
                new_direction,
                rz_new,
                rhs,
            ), residual_norm

        # 初期状態の計算
        def compute_initial_state():
            """初期状態を計算"""
            residual = rhs_array - laplacian_operator(solution)
            direction = preconditioner(residual)  # 前処理付き初期方向
            rz_old = dot_product(residual, direction)
            return solution, residual, direction, rz_old, rhs_array

        # メインソルバーループ
        @jax.jit
        def solve_cg(initial_state):
            """共役勾配法のメインループ"""

            def iteration_condition(state_and_iteration):
                """反復の停止条件を判定"""
                state, iteration = state_and_iteration
                solution, residual, *_ = state

                # 残差の相対ノルムを計算
                relative_residual = jnp.linalg.norm(residual) / (
                    jnp.linalg.norm(rhs_array) + 1e-15
                )

                # 収束判定
                converged = jnp.logical_or(
                    relative_residual <= self.config.tolerance,
                    iteration >= self.config.max_iterations - 1,
                )

                return converged

            def iteration_body(state_and_iteration):
                """単一の反復ステップ"""
                state, iteration = state_and_iteration

                # CG反復の実行
                new_state, _ = cg_iteration(state)

                return new_state, iteration + 1

            # 初期状態と反復回数
            initial_loop_state = (initial_state, jnp.array(0, dtype=jnp.int32))

            # メインの反復ループ
            final_state, num_iterations = lax.while_loop(
                iteration_condition, iteration_body, initial_loop_state
            )

            # 最終的な解を返す
            solution, *_ = final_state
            return solution, num_iterations

        # ソルバーの実行
        solution, iterations = solve_cg(compute_initial_state())

        # 状態の更新
        self._iteration_count = int(iterations)
        self._converged = True
        self._error_history = []

        return solution

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "solver_type": "Conjugate Gradient",
                "final_residual": self._error_history[-1]
                if self._error_history
                else None,
                "iterations": self._iteration_count,
            }
        )
        return diag
