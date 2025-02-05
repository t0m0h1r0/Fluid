import numpy as np
from typing import Optional
from .base import PoissonSolver
from ...field.scalar_field import ScalarField

class JacobiSolver(PoissonSolver):
    """Jacobi法によるPoisson方程式ソルバー"""
    
    def __init__(self, omega: float = 0.8, **kwargs):
        """
        Args:
            omega: 緩和係数 (0 < ω < 2)
        """
        super().__init__(**kwargs)
        if not 0 < omega < 2:
            raise ValueError("緩和係数は0から2の間である必要があります")
        self.omega = omega

    def solve(self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None) -> ScalarField:
        dx = rhs.dx
        dx2 = [d*d for d in dx]
        
        # 解の初期化
        if initial_guess is None:
            solution = ScalarField(rhs.metadata)
        else:
            solution = initial_guess
        
        phi = solution.data
        f = rhs.data
        
        # メインの反復
        self.iteration_count = 0
        self.residual_history = []
        
        while self.iteration_count < self.max_iterations:
            phi_new = np.zeros_like(phi)
            
            # Jacobiの反復
            for i in range(1, phi.shape[0]-1):
                for j in range(1, phi.shape[1]-1):
                    for k in range(1, phi.shape[2]-1):
                        phi_new[i,j,k] = (
                            (phi[i+1,j,k] + phi[i-1,j,k])/dx2[0] +
                            (phi[i,j+1,k] + phi[i,j-1,k])/dx2[1] +
                            (phi[i,j,k+1] + phi[i,j,k-1])/dx2[2] -
                            f[i,j,k]
                        ) / (2/dx2[0] + 2/dx2[1] + 2/dx2[2])
            
            # SOR緩和
            phi = (1 - self.omega) * phi + self.omega * phi_new
            
            # 収束判定
            residual = self.compute_residual(solution, rhs)
            self.residual_history.append(residual)
            
            if self.has_converged():
                break
                
            self.iteration_count += 1
        
        solution.data = phi
        return solution