"""
マルチグリッド法によるPoisson方程式の高速解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField


class PoissonMultigridSolver(PoissonSolverBase):
    """マルチグリッド法によるPoissonソルバー（JAX最適化版）"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        cycle_type: str = "V",
        num_levels: int = 3,
    ):
        """
        Args:
            config: ソルバーの設定
            cycle_type: マルチグリッドサイクルの種類 ('V', 'W')
            num_levels: グリッドの階層数
        """
        super().__init__(config)
        self.cycle_type = cycle_type
        self.num_levels = num_levels
        self._init_multigrid_operators()

    def _init_multigrid_operators(self):
        """マルチグリッド法に特化したJAX最適化演算子を初期化"""

        @partial(jit, static_argnums=(0,))
        def restrict(self, fine_grid: jnp.ndarray) -> jnp.ndarray:
            """制限演算子（JAX最適化版）"""
            # パディングを使用して2x2x2ブロックの平均を計算
            padded = jnp.pad(fine_grid, 1, mode="edge")

            # vmapを使用して効率的に平均を計算
            average_op = vmap(vmap(vmap(lambda x: jnp.mean(x.reshape(-1)))))
            blocks = jnp.lib.stride_tricks.sliding_window_view(padded, (2, 2, 2))

            return average_op(blocks[::2, ::2, ::2])

        @partial(jit, static_argnums=(0,))
        def prolongate(
            self, coarse_grid: jnp.ndarray, fine_shape: Tuple[int, ...]
        ) -> jnp.ndarray:
            """補間演算子（JAX最適化版）"""
            # 直接補間
            fine_grid = jnp.zeros(fine_shape)

            # 基本点の設定
            fine_grid = fine_grid.at[::2, ::2, ::2].set(coarse_grid)

            # 面の補間
            fine_grid = fine_grid.at[1::2, ::2, ::2].set(
                (fine_grid[:-1:2, ::2, ::2] + fine_grid[2::2, ::2, ::2]) / 2
            )
            fine_grid = fine_grid.at[::2, 1::2, ::2].set(
                (fine_grid[::2, :-1:2, ::2] + fine_grid[::2, 2::2, ::2]) / 2
            )
            fine_grid = fine_grid.at[::2, ::2, 1::2].set(
                (fine_grid[::2, ::2, :-1:2] + fine_grid[::2, ::2, 2::2]) / 2
            )

            # エッジの補間
            fine_grid = fine_grid.at[1::2, 1::2, ::2].set(
                (fine_grid[1::2, :-1:2, ::2] + fine_grid[1::2, 2::2, ::2]) / 2
            )
            fine_grid = fine_grid.at[1::2, ::2, 1::2].set(
                (fine_grid[1::2, ::2, :-1:2] + fine_grid[1::2, ::2, 2::2]) / 2
            )
            fine_grid = fine_grid.at[::2, 1::2, 1::2].set(
                (fine_grid[::2, 1::2, :-1:2] + fine_grid[::2, 1::2, 2::2]) / 2
            )

            return fine_grid

        @partial(jit, static_argnums=(0,))
        def smooth(
            self, solution: jnp.ndarray, rhs: jnp.ndarray, num_iterations: int = 2
        ) -> jnp.ndarray:
            """スムーサー（JAX最適化版）"""

            def single_iteration(solution, _):
                dx2 = jnp.square(self.config.dx)

                # 各方向の隣接点からの寄与
                neighbors_sum = (
                    (jnp.roll(solution, 1, axis=0) + jnp.roll(solution, -1, axis=0))
                    / dx2[0]
                    + (jnp.roll(solution, 1, axis=1) + jnp.roll(solution, -1, axis=1))
                    / dx2[1]
                    + (jnp.roll(solution, 1, axis=2) + jnp.roll(solution, -1, axis=2))
                    / dx2[2]
                )

                # 係数の計算
                coeff = 2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2])

                # 更新
                return (neighbors_sum - rhs) / coeff, None

            smoothed, _ = lax.scan(
                single_iteration, solution, None, length=num_iterations
            )
            return smoothed

        @partial(jit, static_argnums=(0,))
        def compute_residual(
            self, solution: jnp.ndarray, rhs: jnp.ndarray
        ) -> jnp.ndarray:
            """残差の計算（JAX最適化版）"""
            return rhs - self.laplacian_operator(solution)

        self.restrict = restrict
        self.prolongate = prolongate
        self.smooth = smooth
        self.compute_residual = compute_residual

    @partial(jit, static_argnums=(0,))
    def _multigrid_cycle(
        self, solution: jnp.ndarray, rhs: jnp.ndarray, level: int
    ) -> jnp.ndarray:
        """マルチグリッドサイクル（JAX最適化版）"""

        def solve_at_level(state):
            solution, rhs, level = state

            def base_case(_):
                return self.smooth(solution, rhs, num_iterations=50)

            def recursive_case(_):
                # 前平滑化
                smoothed = self.smooth(solution, rhs)

                # 残差の計算と制限
                residual = self.compute_residual(smoothed, rhs)
                coarse_residual = self.restrict(residual)

                # 粗いグリッドでの補正を計算
                coarse_shape = tuple(s // 2 for s in smoothed.shape)
                coarse_correction = jnp.zeros(coarse_shape)

                # 再帰的に解く
                coarse_solution = solve_at_level(
                    (coarse_correction, coarse_residual, level + 1)
                )

                # 補正を補間して加える
                correction = self.prolongate(coarse_solution, smoothed.shape)
                corrected = smoothed + correction

                # 後平滑化
                return self.smooth(corrected, rhs)

            # レベルに応じて処理を分岐
            return lax.cond(
                level == self.num_levels - 1, base_case, recursive_case, None
            )

        # 初期状態から解を計算
        return solve_at_level((solution, rhs, level))

    def solve(
        self,
        rhs: Union[np.ndarray, jnp.ndarray, ScalarField],
        initial_guess: Optional[Union[np.ndarray, jnp.ndarray, ScalarField]] = None,
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
        if initial_array is None:
            solution = jnp.zeros_like(rhs_array)
        else:
            solution = initial_array

        def iteration_step(carry):
            solution, iteration = carry

            # マルチグリッドサイクルの実行
            new_solution = self._multigrid_cycle(solution, rhs_array, 0)

            # 収束判定のための残差計算
            residual = jnp.max(jnp.abs(new_solution - solution))
            self._error_history.append(float(residual))

            return (new_solution, iteration + 1), residual

        def convergence_test(carry_and_residual):
            (_, iteration), residual = carry_and_residual
            return jnp.logical_and(
                iteration < self.config.max_iterations, residual > self.config.tolerance
            )

        # メインループの実行
        (solution, self._iteration_count), final_residual = lax.while_loop(
            convergence_test, lambda x: iteration_step(x[0]), ((solution, 0), jnp.inf)
        )

        self._converged = final_residual <= self.config.tolerance
        return solution

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "solver_type": "Multigrid",
                "cycle_type": self.cycle_type,
                "num_levels": self.num_levels,
                "final_residual": self._error_history[-1]
                if self._error_history
                else None,
            }
        )
        return diag
