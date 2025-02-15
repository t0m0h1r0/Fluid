"""
逐次過緩和法（SOR: Successive Over-Relaxation）による
Poisson方程式の解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union
import numpy as np
import jax.numpy as jnp
from jax import jit

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
        pass  # 初期化は solve メソッド内で動的に行う

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

        # グリッド間隔と緩和パラメータの取得
        dx2 = jnp.square(jnp.array(self.config.dx))
        omega = self.config.relaxation_parameter

        # メインループ用の状態を定義
        initial_state = {"solution": solution, "residual": jnp.inf, "step_count": 0}

        # JAX最適化された SOR 更新関数
        @jit
        def sor_update_and_check(state):
            sol = state["solution"]

            def update_solution(sol):
                for i in range(1, sol.shape[0] - 1):
                    for j in range(1, sol.shape[1] - 1):
                        for k in range(1, sol.shape[2] - 1):
                            # 隣接点からの寄与
                            neighbors_sum = (
                                (sol[i + 1, j, k] + sol[i - 1, j, k]) / dx2[0]
                                + (sol[i, j + 1, k] + sol[i, j - 1, k]) / dx2[1]
                                + (sol[i, j, k + 1] + sol[i, j, k - 1]) / dx2[2]
                            )

                            # 係数の計算
                            coeff = 2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2])

                            # SOR更新
                            new_value = (1 - omega) * sol[i, j, k] + (omega / coeff) * (
                                neighbors_sum - rhs_array[i, j, k]
                            )

                            sol = sol.at[i, j, k].set(new_value)
                return sol

            # ソリューションの更新
            new_solution = update_solution(sol)

            # 残差の計算（収束判定用）
            residual = jnp.max(jnp.abs(new_solution - sol))

            # 状態の更新
            return {
                "solution": new_solution,
                "residual": residual,
                "step_count": state["step_count"] + 1,
            }

        # 停止条件
        def cond_fun(state):
            return jnp.logical_and(
                state["residual"] > self.config.tolerance,
                state["step_count"] < self.config.max_iterations,
            )

        # メインループ
        result_state = initial_state
        while cond_fun(result_state):
            result_state = sor_update_and_check(result_state)

        # ステップ数と収束状態の更新
        solution = result_state["solution"]
        final_residual = result_state["residual"]
        self._iteration_count = result_state["step_count"]
        self._converged = final_residual <= self.config.tolerance
        self._error_history.append(float(final_residual))

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
