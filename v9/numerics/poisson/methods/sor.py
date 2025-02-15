"""
逐次過緩和法（SOR: Successive Over-Relaxation）による
Poisson方程式の解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField


class PoissonSORSolver(PoissonSolverBase):
    """SOR法によるPoissonソルバー（JAX最適化版）"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
    ):
        """
        Args:
            config: ソルバーの設定
        """
        super().__init__(config)
        self._init_sor_operators()

    def _init_sor_operators(self):
        """SOR法に特化したJAX最適化演算子を初期化"""

        @partial(jit, static_argnums=(0,))
        def sor_update(self, solution: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
            dx2 = jnp.square(self.config.dx)
            omega = self.config.relaxation_parameter

            def body_fun(i, sol):
                def inner_body_fun(j, sol_i):
                    def innermost_body_fun(k, sol_ij):
                        # 隣接点からの寄与
                        neighbors_sum = (
                            (sol_ij[i + 1, j, k] + sol_ij[i - 1, j, k]) / dx2[0]
                            + (sol_ij[i, j + 1, k] + sol_ij[i, j - 1, k]) / dx2[1]
                            + (sol_ij[i, j, k + 1] + sol_ij[i, j, k - 1]) / dx2[2]
                        )

                        # 係数の計算
                        coeff = 2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2])

                        # SOR更新
                        new_value = (1 - omega) * sol_ij[i, j, k] + (omega / coeff) * (
                            neighbors_sum - rhs[i, j, k]
                        )

                        return sol_ij.at[i, j, k].set(new_value)

                    return lax.fori_loop(
                        1, sol_i.shape[2] - 1, innermost_body_fun, sol_i
                    )

                return lax.fori_loop(1, sol.shape[1] - 1, inner_body_fun, sol)

            return lax.fori_loop(1, solution.shape[0] - 1, body_fun, solution)

        @partial(jit, static_argnums=(0,))
        def compute_residual(self, solution: jnp.ndarray, rhs: jnp.ndarray) -> float:
            """残差ノルムを計算（JAX最適化版）"""
            residual = self.laplacian_operator(solution) - rhs
            return jnp.max(jnp.abs(residual))

        self.sor_update = sor_update
        self.compute_residual = compute_residual

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

        # SORの反復
        def sor_iteration(carry):
            solution, step = carry
            new_solution = self.sor_update(solution, rhs_array)
            residual = self.compute_residual(new_solution, rhs_array)
            self._error_history.append(float(residual))
            return (new_solution, step + 1), residual

        def cond_fun(carry_and_residual):
            _, step = carry_and_residual[0]
            residual = carry_and_residual[1]
            return jnp.logical_and(
                step < self.config.max_iterations, residual > self.config.tolerance
            )

        # メインループ
        (solution, step), final_residual = lax.while_loop(
            cond_fun, lambda x: sor_iteration(x[0]), ((solution, 0), jnp.inf)
        )

        self._iteration_count = int(step)
        self._converged = final_residual <= self.config.tolerance

        return solution

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "solver_type": "SOR",
                "relaxation_parameter": self.config.relaxation_parameter,
                "final_residual": self._error_history[-1]
                if self._error_history
                else None,
            }
        )
        return diag
