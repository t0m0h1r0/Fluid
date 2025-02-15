from typing import Optional, Dict, Any, Union
import numpy as np
import jax.numpy as jnp
from jax import jit, lax

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
        solution = jnp.zeros_like(rhs_array) if initial_array is None else initial_array

        # グリッド間隔と緩和パラメータの取得
        dx2 = jnp.square(jnp.array(self.config.dx))
        omega = self.config.relaxation_parameter
        coeff = 2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2])

        def sor_step(sol):
            """SOR更新ステップをJAX最適化"""

            def update_fn(index, sol):
                i, j, k = index
                neighbors_sum = (
                    (sol[i + 1, j, k] + sol[i - 1, j, k]) / dx2[0]
                    + (sol[i, j + 1, k] + sol[i, j - 1, k]) / dx2[1]
                    + (sol[i, j, k + 1] + sol[i, j, k - 1]) / dx2[2]
                )
                new_value = (1 - omega) * sol[i, j, k] + (omega / coeff) * (
                    neighbors_sum - rhs_array[i, j, k]
                )
                return sol.at[i, j, k].set(new_value)

            indices = jnp.array(
                [
                    (i, j, k)
                    for i in range(1, sol.shape[0] - 1)
                    for j in range(1, sol.shape[1] - 1)
                    for k in range(1, sol.shape[2] - 1)
                ],
                dtype=jnp.int32,
            )
            return lax.fori_loop(
                0, indices.shape[0], lambda idx, sol: update_fn(indices[idx], sol), sol
            )

        @jit
        def iterate(state):
            new_solution = sor_step(state["solution"])
            residual = jnp.max(jnp.abs(new_solution - state["solution"]))
            return {
                "solution": new_solution,
                "residual": residual,
                "step_count": state["step_count"] + 1,
            }

        result_state = {"solution": solution, "residual": jnp.inf, "step_count": 0}

        def cond_fun(state):
            return jnp.logical_and(
                state["residual"] > self.config.tolerance,
                state["step_count"] < self.config.max_iterations,
            )

        while cond_fun(result_state):
            result_state = iterate(result_state)

        # ステップ数と収束状態の更新
        self._iteration_count = result_state["step_count"]
        self._converged = result_state["residual"] <= self.config.tolerance
        self._error_history.append(float(result_state["residual"]))

        print(
            f"Final Step {self._iteration_count}: Residual = {result_state['residual']}"
        )

        return result_state["solution"]

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
