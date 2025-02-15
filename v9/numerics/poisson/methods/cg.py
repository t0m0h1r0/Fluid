"""
共役勾配法によるPoisson方程式の数値解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union, Tuple
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

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
        super().__init__(config)
        self._init_cg_operators()

    def _init_cg_operators(self):
        """CG法に特化したJAX最適化演算子を初期化"""

        @partial(jit, static_argnums=(0,))
        def dot_product(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
            """ベクトルの内積を計算（JAX最適化版）"""
            return jnp.sum(x * y)

        @partial(jit, static_argnums=(0,))
        def cg_iteration(
            self,
            state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float],
            rhs: jnp.ndarray,
        ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], float]:
            """単一のCG反復（JAX最適化版）"""
            solution, residual, direction, rz_old = state

            # Ap の計算
            Ap = self.laplacian_operator(direction)

            # ステップサイズの計算
            alpha = rz_old / (self.dot_product(direction, Ap) + 1e-15)

            # 解と残差の更新
            new_solution = solution + alpha * direction
            new_residual = residual - alpha * Ap

            # 新しい内積の計算
            rz_new = self.dot_product(new_residual, new_residual)

            # 方向ベクトルの更新
            beta = rz_new / (rz_old + 1e-15)
            new_direction = new_residual + beta * direction

            # 残差ノルムの計算
            residual_norm = jnp.sqrt(rz_new)

            return (new_solution, new_residual, new_direction, rz_new), residual_norm

        self.dot_product = dot_product
        self.cg_iteration = cg_iteration

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

        # 初期残差の計算
        residual = rhs_array - self.laplacian_operator(solution)
        direction = residual.copy()
        rz_old = self.dot_product(residual, residual)

        # 初期状態の設定
        initial_state = (solution, residual, direction, rz_old)

        # メインループの条件
        def cond_fun(carry_and_residual):
            _, step = carry_and_residual[0]
            residual_norm = carry_and_residual[1]
            return jnp.logical_and(
                step < self.config.max_iterations, residual_norm > self.config.tolerance
            )

        # 反復計算
        def body_fun(carry_and_residual):
            state, step = carry_and_residual[0]
            new_state, residual_norm = self.cg_iteration(state, rhs_array)
            self._error_history.append(float(residual_norm))
            return ((new_state, step + 1), residual_norm)

        # メインループの実行
        (final_state, step), final_residual = lax.while_loop(
            cond_fun, body_fun, ((initial_state, 0), jnp.inf)
        )

        self._iteration_count = int(step)
        self._converged = final_residual <= self.config.tolerance

        return final_state[0]  # solution

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "solver_type": "Conjugate Gradient",
                "final_residual": self._error_history[-1]
                if self._error_history
                else None,
            }
        )
        return diag
