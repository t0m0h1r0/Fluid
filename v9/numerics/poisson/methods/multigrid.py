"""
マルチグリッド法によるPoisson方程式の高速解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union, Tuple
import jax
import jax.numpy as jnp
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
        
        # JAXの最適化設定
        self._jax_config()
        self._init_multigrid_operators()

    def _jax_config(self):
        """JAXの最適化設定"""
        # 64ビット浮動小数点数を使用（高精度計算）
        jax.config.update('jax_enable_x64', True)

    def _init_multigrid_operators(self):
        """マルチグリッド法に特化したJAX最適化演算子を初期化"""

        @partial(jax.jit, static_argnums=(1,))
        def restrict(fine_grid: jnp.ndarray, shape: Tuple[int, int, int]) -> jnp.ndarray:
            """制限演算子（JAX最適化版）"""
            coarse_shape = tuple(s // 2 for s in shape)
            
            # 3D配列のブロック平均を計算
            result = jnp.zeros(coarse_shape)
            for i in range(coarse_shape[0]):
                for j in range(coarse_shape[1]):
                    for k in range(coarse_shape[2]):
                        block = fine_grid[
                            2*i:2*i+2, 
                            2*j:2*j+2, 
                            2*k:2*k+2
                        ]
                        result = result.at[i, j, k].set(jnp.mean(block))
            return result

        @partial(jax.jit, static_argnums=(1,))
        def prolongate(
            coarse_grid: jnp.ndarray, 
            fine_shape: Tuple[int, int, int]
        ) -> jnp.ndarray:
            """補間演算子（JAX最適化版）"""
            fine_grid = jnp.zeros(fine_shape)

            # 直接点の補間
            fine_grid = fine_grid.at[::2, ::2, ::2].set(coarse_grid)

            # エッジの線形補間
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

        @partial(jax.jit, static_argnums=(3,))
        def smooth(
            solution: jnp.ndarray, 
            rhs: jnp.ndarray, 
            dx: jnp.ndarray, 
            is_coarse_grid: bool
        ) -> jnp.ndarray:
            """スムーサー（JAX最適化版）"""
            num_iterations = 50 if is_coarse_grid else 2
            dx2 = dx * dx

            def update_solution(u):
                """単一の反復更新"""
                new_u = u.copy()
                
                # 内部点の更新
                for i in range(1, u.shape[0] - 1):
                    for j in range(1, u.shape[1] - 1):
                        for k in range(1, u.shape[2] - 1):
                            # 隣接点からの寄与
                            neighbors_sum = (
                                (u[i + 1, j, k] + u[i - 1, j, k]) / dx2[0]
                                + (u[i, j + 1, k] + u[i, j - 1, k]) / dx2[1]
                                + (u[i, j, k + 1] + u[i, j, k - 1]) / dx2[2]
                            )
                            
                            # 係数の計算
                            coeff = 2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2])
                            
                            # SOR更新
                            new_u = new_u.at[i, j, k].set(
                                (1 - 1.5) * u[i, j, k] 
                                + (1.5 / coeff) * (neighbors_sum - rhs[i, j, k])
                            )
                
                return new_u

            # 固定回数の反復
            def body_fun(_, x):
                return update_solution(x)
            
            return jax.lax.fori_loop(0, num_iterations, body_fun, solution)

        # オペレータの設定
        self.restrict = restrict
        self.prolongate = prolongate
        self.smooth = smooth

        # ラプラシアンのJIT最適化
        self.laplacian_operator = jax.jit(self.laplacian_operator)

    def solve(
        self,
        rhs: Union[jnp.ndarray, ScalarField],
        initial_guess: Optional[Union[jnp.ndarray, ScalarField]] = None,
    ) -> jnp.ndarray:
        """Poisson方程式を解く"""
        # 入力の検証と配列への変換
        rhs_array, initial_array = self.validate_input(rhs, initial_guess)

        # 初期解の準備
        solution = initial_array if initial_array is not None else jnp.zeros_like(rhs_array)

        # グリッド間隔
        dx = jnp.array(self.config.dx)

        def multigrid_cycle(solution, rhs, level):
            """単一のマルチグリッドサイクル（JIT最適化）"""
            # 最も粗いレベルの処理
            if level == self.num_levels - 1:
                return self.smooth(solution, rhs, dx, is_coarse_grid=True)
            
            # 通常のマルチグリッドサイクル
            # 前平滑化
            solution = self.smooth(solution, rhs, dx, is_coarse_grid=False)

            # 残差の計算
            residual = rhs - self.laplacian_operator(solution)

            # 粗いグリッドへの制限
            coarse_residual = self.restrict(residual, residual.shape)
            coarse_solution = jnp.zeros_like(coarse_residual)

            # 粗いグリッドでの解法
            coarse_solution = multigrid_cycle(coarse_solution, coarse_residual, level + 1)

            # 補正の補間と適用
            correction = self.prolongate(coarse_solution, solution.shape)
            solution = solution + correction

            # 後平滑化
            solution = self.smooth(solution, rhs, dx, is_coarse_grid=False)

            return solution

        # ソルバーループ関数
        def solver_iteration(state):
            """単一のソルバー反復"""
            solution, best_solution, best_residual = state
            
            # マルチグリッドサイクルの適用
            solution = multigrid_cycle(solution, rhs_array, 0)
            
            # 残差の計算
            residual = jnp.max(jnp.abs(rhs_array - self.laplacian_operator(solution)))
            
            # より良い解の選択
            new_best_solution = jax.lax.cond(
                residual < best_residual,
                lambda _: solution,
                lambda _: best_solution
            )
            new_best_residual = jnp.minimum(residual, best_residual)
            
            return solution, new_best_solution, new_best_residual

        # 初期状態の設定
        initial_state = (solution, solution, jnp.inf)
        
        # メインソルバーループ
        final_state = jax.lax.fori_loop(
            0, self.config.max_iterations, 
            lambda _, state: solver_iteration(state), 
            initial_state
        )
        
        # 最終的な解と残差の取得
        solution, best_solution, final_residual = final_state

        # 状態の更新
        self._iteration_count = self.config.max_iterations
        self._converged = final_residual < self.config.tolerance
        self._error_history = [float(final_residual)]

        return best_solution

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