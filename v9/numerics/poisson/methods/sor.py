"""
逐次過緩和法（SOR: Successive Over-Relaxation）による
Poisson方程式の解法
"""

from typing import Optional, Dict, Any
import numpy as np
import jax.numpy as jnp

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField


class PoissonSORSolver(PoissonSolverBase):
    """SOR法によるPoissonソルバー"""

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
        rhs: np.ndarray | jnp.ndarray | ScalarField,
        initial_guess: Optional[np.ndarray | jnp.ndarray | ScalarField] = None,
    ) -> np.ndarray:
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
            solution = np.zeros_like(rhs_array)
        else:
            solution = initial_array.copy()

        # SORの反復計算
        dx = self.config.dx
        omega = self.config.relaxation_parameter
        max_iter = self.config.max_iterations
        tol = self.config.tolerance

        # グリッド間隔の二乗の逆数を計算
        idx2 = 1.0 / (dx * dx)

        # 反復計算
        for iter_count in range(max_iter):
            old_solution = solution.copy()

            # 3次元格子点でのSOR反復
            for i in range(1, solution.shape[0] - 1):
                for j in range(1, solution.shape[1] - 1):
                    for k in range(1, solution.shape[2] - 1):
                        # 隣接点からの寄与
                        sum_neighbors = (
                            (solution[i + 1, j, k] + solution[i - 1, j, k]) * idx2[0]
                            + (solution[i, j + 1, k] + solution[i, j - 1, k]) * idx2[1]
                            + (solution[i, j, k + 1] + solution[i, j, k - 1]) * idx2[2]
                        )

                        # ラプラシアン係数の計算
                        coeff = 2.0 * (idx2[0] + idx2[1] + idx2[2])

                        # SOR更新
                        solution[i, j, k] = (1 - omega) * solution[i, j, k] + (
                            omega / coeff
                        ) * (sum_neighbors - rhs_array[i, j, k])

            # 収束判定
            residual = np.max(np.abs(solution - old_solution))
            self._error_history.append(float(residual))

            if residual < tol:
                self._converged = True
                break

        self._iteration_count = iter_count + 1

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
