from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Tuple
from core.field.scalar_field import ScalarField

class PoissonSolver(ABC):
    """Poisson方程式ソルバーの基底クラス"""
    
    def __init__(self, 
                 max_iterations: int = 1000, 
                 tolerance: float = 1e-6,
                 **kwargs):  # 追加のパラメータを許容
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.iteration_count = 0
        self.residual_history: List[float] = []

    @abstractmethod
    def solve(self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None) -> ScalarField:
        """Poisson方程式を解く
        ∇²φ = f
        
        Args:
            rhs: 右辺のソース項
            initial_guess: 初期推定値
        Returns:
            解φ
        """
        pass

    def compute_residual(self, solution: ScalarField, rhs: ScalarField) -> float:
        """残差 ||∇²φ - f||を計算"""
        dx = solution.dx
        laplacian = np.zeros_like(solution.data)
        
        for i in range(3):
            laplacian += np.gradient(
                np.gradient(solution.data, dx[i], axis=i),
                dx[i], axis=i
            )
        
        residual = laplacian - rhs.data
        return np.sqrt(np.mean(residual**2))

    def has_converged(self) -> bool:
        """収束判定"""
        if not self.residual_history:
            return False
        return self.residual_history[-1] < self.tolerance

    def get_convergence_info(self) -> Tuple[int, List[float]]:
        """収束情報の取得"""
        return self.iteration_count, self.residual_history