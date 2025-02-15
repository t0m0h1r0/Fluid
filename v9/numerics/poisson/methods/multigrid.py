"""
マルチグリッド法によるPoisson方程式の高速解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import jax.numpy as jnp
from jax import jit

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

        @jit
        def restrict(fine_grid: jnp.ndarray) -> jnp.ndarray:
            """制限演算子（JAX最適化版）"""
            shape = fine_grid.shape
            coarse_shape = tuple(s // 2 for s in shape)
            result = jnp.zeros(coarse_shape)

            for i in range(coarse_shape[0]):
                for j in range(coarse_shape[1]):
                    for k in range(coarse_shape[2]):
                        result = result.at[i, j, k].set(
                            jnp.mean(
                                fine_grid[
                                    2 * i : 2 * i + 2,
                                    2 * j : 2 * j + 2,
                                    2 * k : 2 * k + 2,
                                ]
                            )
                        )
            return result

        @jit
        def prolongate(
            coarse_grid: jnp.ndarray, fine_shape: Tuple[int, ...]
        ) -> jnp.ndarray:
            """補間演算子（JAX最適化版）"""
            fine_grid = jnp.zeros(fine_shape)

            # 直接点の補間
            fine_grid = fine_grid.at[::2, ::2, ::2].set(coarse_grid)

            # エッジの補間
            fine_grid = fine_grid.at[1::2, ::2, ::2].set(
                (fine_grid[:-1:2, ::2, ::2] + fine_grid[2::2, ::2, ::2]) / 2
            )
            fine_grid = fine_grid.at[::2, 1::2, ::2].set(
                (fine_grid[::2, :-1:2, ::2] + fine_grid[::2, 2::2, ::2]) / 2
            )
            fine_grid = fine_grid.at[::2, ::2, 1::2].set(
                (fine_grid[::2, ::2, :-1:2] + fine_grid[::2, ::2, 2::2]) / 2
            )

            return fine_grid

        # スムーサーのための静的関数を定義
        def smooth_fine_grid(
            solution: jnp.ndarray, rhs: jnp.ndarray, dx: jnp.ndarray
        ) -> jnp.ndarray:
            """細かいグリッド用のスムーサー（2回反復）"""
            dx2 = dx * dx

            def single_iteration(u):
                for i in range(1, u.shape[0] - 1):
                    for j in range(1, u.shape[1] - 1):
                        for k in range(1, u.shape[2] - 1):
                            u = u.at[i, j, k].set(
                                (
                                    (u[i + 1, j, k] + u[i - 1, j, k]) / dx2[0]
                                    + (u[i, j + 1, k] + u[i, j - 1, k]) / dx2[1]
                                    + (u[i, j, k + 1] + u[i, j, k - 1]) / dx2[2]
                                    - rhs[i, j, k]
                                )
                                / (2 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2]))
                            )
                return u

            # 2回の反復
            solution = single_iteration(solution)
            solution = single_iteration(solution)
            return solution

        def smooth_coarse_grid(
            solution: jnp.ndarray, rhs: jnp.ndarray, dx: jnp.ndarray
        ) -> jnp.ndarray:
            """粗いグリッド用のスムーサー（50回反復）"""
            dx2 = dx * dx

            def single_iteration(u):
                for i in range(1, u.shape[0] - 1):
                    for j in range(1, u.shape[1] - 1):
                        for k in range(1, u.shape[2] - 1):
                            u = u.at[i, j, k].set(
                                (
                                    (u[i + 1, j, k] + u[i - 1, j, k]) / dx2[0]
                                    + (u[i, j + 1, k] + u[i, j - 1, k]) / dx2[1]
                                    + (u[i, j, k + 1] + u[i, j, k - 1]) / dx2[2]
                                    - rhs[i, j, k]
                                )
                                / (2 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2]))
                            )
                return u

            # 50回の反復
            for _ in range(50):
                solution = single_iteration(solution)
            return solution

        # JAX最適化された静的スムーサー
        smooth_fine_grid_jit = jit(smooth_fine_grid)
        smooth_coarse_grid_jit = jit(smooth_coarse_grid)

        # インスタンスメソッドに割り当て
        def smooth(solution, rhs, dx, is_coarse_grid=False):
            """スムーサーのラッパー関数"""
            if is_coarse_grid:
                return smooth_coarse_grid_jit(solution, rhs, dx)
            else:
                return smooth_fine_grid_jit(solution, rhs, dx)

        # 演算子の設定
        self.restrict = restrict
        self.prolongate = prolongate
        self.smooth = smooth

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

        # グリッド間隔
        dx = jnp.array(self.config.dx)

        # マルチグリッドサイクル関数
        def multigrid_cycle(solution, rhs, level):
            """単一のマルチグリッドサイクル"""
            # 最も粗いレベルでの解法
            if level == self.num_levels - 1:
                return self.smooth(solution, rhs, dx, is_coarse_grid=True)

            # 前平滑化
            solution = self.smooth(solution, rhs, dx, is_coarse_grid=False)

            # 残差の計算
            residual = rhs - self.laplacian_operator(solution)

            # 粗いグリッドへの制限
            coarse_residual = self.restrict(residual)
            coarse_solution = jnp.zeros_like(coarse_residual)

            # 粗いグリッドでの解法
            coarse_solution = multigrid_cycle(
                coarse_solution, coarse_residual, level + 1
            )

            # 補正の補間と適用
            correction = self.prolongate(coarse_solution, solution.shape)
            solution = solution + correction

            # 後平滑化
            solution = self.smooth(solution, rhs, dx, is_coarse_grid=False)

            return solution

        # マルチグリッドサイクルの反復
        solution_history = [solution]
        residual_history = []

        for iteration in range(self.config.max_iterations):
            # マルチグリッドサイクルの適用
            solution = multigrid_cycle(solution, rhs_array, 0)
            solution_history.append(solution)

            # 残差の計算
            residual = jnp.max(jnp.abs(rhs_array - self.laplacian_operator(solution)))
            residual_history.append(float(residual))

            # 収束判定
            if residual < self.config.tolerance:
                self._converged = True
                break

        # 最終的な状態の更新
        self._iteration_count = iteration + 1
        self._error_history = residual_history

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
