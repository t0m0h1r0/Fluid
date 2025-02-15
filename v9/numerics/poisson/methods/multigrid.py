"""
マルチグリッド法によるPoisson方程式の高速解法
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import jax.numpy as jnp

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField


class PoissonMultigridSolver(PoissonSolverBase):
    """マルチグリッド法によるPoissonソルバー"""

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

    def _restrict(self, fine_grid: np.ndarray) -> np.ndarray:
        """細かいグリッドから粗いグリッドへの制限"""
        nx, ny, nz = fine_grid.shape
        coarse_grid = np.zeros((nx//2, ny//2, nz//2))

        for i in range(nx//2):
            for j in range(ny//2):
                for k in range(nz//2):
                    coarse_grid[i,j,k] = np.mean(
                        fine_grid[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2]
                    )

        return coarse_grid

    def _prolongate(self, coarse_grid: np.ndarray, fine_shape: Tuple[int, ...]) -> np.ndarray:
        """粗いグリッドから細かいグリッドへの補間"""
        fine_grid = np.zeros(fine_shape)
        
        for i in range(coarse_grid.shape[0]):
            for j in range(coarse_grid.shape[1]):
                for k in range(coarse_grid.shape[2]):
                    # 直接点の補間
                    fine_grid[2*i, 2*j, 2*k] = coarse_grid[i,j,k]
                    
                    # エッジの補間
                    if 2*i+1 < fine_shape[0]:
                        fine_grid[2*i+1, 2*j, 2*k] = 0.5 * (
                            coarse_grid[i,j,k] +
                            (coarse_grid[i+1,j,k] if i+1 < coarse_grid.shape[0] else coarse_grid[i,j,k])
                        )
                    if 2*j+1 < fine_shape[1]:
                        fine_grid[2*i, 2*j+1, 2*k] = 0.5 * (
                            coarse_grid[i,j,k] +
                            (coarse_grid[i,j+1,k] if j+1 < coarse_grid.shape[1] else coarse_grid[i,j,k])
                        )
                    if 2*k+1 < fine_shape[2]:
                        fine_grid[2*i, 2*j, 2*k+1] = 0.5 * (
                            coarse_grid[i,j,k] +
                            (coarse_grid[i,j,k+1] if k+1 < coarse_grid.shape[2] else coarse_grid[i,j,k])
                        )

        return fine_grid

    def _smooth(
        self,
        solution: np.ndarray,
        rhs: np.ndarray,
        num_iterations: int = 2
    ) -> np.ndarray:
        """Gauss-Seidelスムーサー"""
        dx = self.config.dx
        idx2 = 1.0 / (dx * dx)
        
        for _ in range(num_iterations):
            for i in range(1, solution.shape[0] - 1):
                for j in range(1, solution.shape[1] - 1):
                    for k in range(1, solution.shape[2] - 1):
                        solution[i,j,k] = (
                            ((solution[i+1,j,k] + solution[i-1,j,k]) * idx2[0] +
                             (solution[i,j+1,k] + solution[i,j-1,k]) * idx2[1] +
                             (solution[i,j,k+1] + solution[i,j,k-1]) * idx2[2] -
                             rhs[i,j,k]) /
                            (2.0 * (idx2[0] + idx2[1] + idx2[2]))
                        )

        return solution

    def _solve_coarse(self, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """最も粗いグリッドでの解法"""
        # 簡単なGauss-Seidel反復
        return self._smooth(solution, rhs, num_iterations=50)

    def _compute_residual(self, solution: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """残差を計算"""
        dx = self.config.dx
        idx2 = 1.0 / (dx * dx)
        
        residual = np.zeros_like(solution)
        for i in range(1, solution.shape[0] - 1):
            for j in range(1, solution.shape[1] - 1):
                for k in range(1, solution.shape[2] - 1):
                    residual[i,j,k] = rhs[i,j,k] - (
                        ((solution[i+1,j,k] + solution[i-1,j,k] - 2*solution[i,j,k]) * idx2[0] +
                         (solution[i,j+1,k] + solution[i,j-1,k] - 2*solution[i,j,k]) * idx2[1] +
                         (solution[i,j,k+1] + solution[i,j,k-1] - 2*solution[i,j,k]) * idx2[2])
                    )
        return residual

    def _multigrid_cycle(
        self,
        solution: np.ndarray,
        rhs: np.ndarray,
        level: int
    ) -> np.ndarray:
        """マルチグリッドサイクルの実行"""
        # 最も粗いレベルでは直接解く
        if level == self.num_levels - 1:
            return self._solve_coarse(solution, rhs)

        # 前平滑化
        solution = self._smooth(solution, rhs)

        # 残差の計算と制限
        residual = self._compute_residual(solution, rhs)
        coarse_residual = self._restrict(residual)
        
        # 粗いグリッドでの補正を計算
        coarse_correction = np.zeros_like(coarse_residual)
        coarse_correction = self._multigrid_cycle(
            coarse_correction,
            coarse_residual,
            level + 1
        )

        # 補正を補間して加える
        correction = self._prolongate(coarse_correction, solution.shape)
        solution += correction

        # 後平滑化
        solution = self._smooth(solution, rhs)

        return solution

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

        # マルチグリッド反復
        for iter_count in range(self.config.max_iterations):
            old_solution = solution.copy()
            
            # マルチグリッドサイクルの実行
            solution = self._multigrid_cycle(solution, rhs_array, 0)

            # 収束判定
            residual = np.max(np.abs(solution - old_solution))
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
            "solver_type": "Multigrid",
            "cycle_type": self.cycle_type,
            "num_levels": self.num_levels,
            "final_residual": self._error_history[-1] if self._error_history else None,
        })
        return diag