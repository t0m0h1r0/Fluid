"""
共役勾配法によるPoisson方程式の数値解法
"""

from typing import Optional, Dict, Any
import numpy as np
import jax.numpy as jnp

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField


class PoissonCGSolver(PoissonSolverBase):
    """共役勾配法によるPoissonソルバー"""

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

        # グリッド間隔の二乗の逆数を計算
        dx = self.config.dx
        idx2 = 1.0 / (dx * dx)

        # ラプラシアン演算子の適用
        def apply_laplacian(x):
            result = np.zeros_like(x)
            for i in range(1, x.shape[0] - 1):
                for j in range(1, x.shape[1] - 1):
                    for k in range(1, x.shape[2] - 1):
                        result[i,j,k] = (
                            (x[i+1,j,k] + x[i-1,j,k] - 2*x[i,j,k]) * idx2[0] +
                            (x[i,j+1,k] + x[i,j-1,k] - 2*x[i,j,k]) * idx2[1] +
                            (x[i,j,k+1] + x[i,j,k-1] - 2*x[i,j,k]) * idx2[2]
                        )
            return result

        # 初期残差の計算
        r = rhs_array - apply_laplacian(solution)
        p = r.copy()

        # ノルムの計算
        def compute_norm(x):
            return np.sqrt(np.sum(x * x))

        # 内積の計算
        def compute_inner_product(x, y):
            return np.sum(x * y)

        # CG反復
        for iter_count in range(self.config.max_iterations):
            Ap = apply_laplacian(p)
            alpha = compute_inner_product(r, r) / (compute_inner_product(p, Ap) + 1e-15)

            # 解の更新
            solution += alpha * p

            # 残差の更新
            r_new = r - alpha * Ap
            beta = compute_inner_product(r_new, r_new) / (compute_inner_product(r, r) + 1e-15)
            r = r_new

            # 探索方向の更新
            p = r + beta * p

            # 収束判定
            residual = compute_norm(r)
            self._error_history.append(float(residual))

            if residual < self.config.tolerance:
                self._converged = True
                break

        self._iteration_count = iter_count + 1

        return solution

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update({
            "solver_type": "Conjugate Gradient",
            "final_residual": self._error_history[-1] if self._error_history else None,
        })
        return diag