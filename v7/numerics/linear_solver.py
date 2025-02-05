from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Tuple

class LinearSolver(ABC):
    """線形ソルバーの基底クラス"""
    def __init__(self, max_iter: int = 1000, tolerance: float = 1e-6):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.iteration_count = 0
        self.residual_history: List[float] = []

    @abstractmethod
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        pass

    def _compute_residual(self, A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
        r = b - A @ x
        return np.linalg.norm(r) / np.linalg.norm(b)

class ConjugateGradient(LinearSolver):
    """共役勾配法"""
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()

        r = b - A @ x
        p = r.copy()
        
        self.iteration_count = 0
        self.residual_history = []
        
        while self.iteration_count < self.max_iter:
            Ap = A @ p
            alpha = np.dot(r, r) / np.dot(p, Ap)
            x += alpha * p
            r_new = r - alpha * Ap
            
            residual = np.linalg.norm(r_new) / np.linalg.norm(b)
            self.residual_history.append(residual)
            
            if residual < self.tolerance:
                break
                
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new
            
            self.iteration_count += 1
            
        return x

class BiCGStab(LinearSolver):
    """BiCGStab法"""
    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()

        r = b - A @ x
        r_tilde = r.copy()
        
        rho_prev = alpha = omega = 1.0
        v = p = np.zeros(n)
        
        self.iteration_count = 0
        self.residual_history = []
        
        while self.iteration_count < self.max_iter:
            rho = np.dot(r_tilde, r)
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)
            v = A @ p
            alpha = rho / np.dot(r_tilde, v)
            h = x + alpha * p
            
            if np.linalg.norm(h - x) < self.tolerance:
                x = h
                break
                
            s = r - alpha * v
            t = A @ s
            omega = np.dot(t, s) / np.dot(t, t)
            x = h + omega * s
            r = s - omega * t
            
            residual = np.linalg.norm(r) / np.linalg.norm(b)
            self.residual_history.append(residual)
            
            if residual < self.tolerance:
                break
                
            rho_prev = rho
            self.iteration_count += 1
            
        return x

class MultiGrid:
    """マルチグリッド法"""
    def __init__(self, 
                 levels: int = 4,
                 v_cycles: int = 3,
                 smoother_iterations: int = 2,
                 tolerance: float = 1e-6):
        self.levels = levels
        self.v_cycles = v_cycles
        self.smoother_iterations = smoother_iterations
        self.tolerance = tolerance
        self.iteration_count = 0
        self.residual_history: List[float] = []

    def solve(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None) -> np.ndarray:
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()

        self.iteration_count = 0
        self.residual_history = []
        
        while self.iteration_count < self.v_cycles:
            x = self._v_cycle(A, b, x, self.levels)
            
            residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
            self.residual_history.append(residual)
            
            if residual < self.tolerance:
                break
                
            self.iteration_count += 1
            
        return x

    def _v_cycle(self, A: np.ndarray, b: np.ndarray, x: np.ndarray, level: int) -> np.ndarray:
        if level == 1:
            # 最粗グリッドでの直接解法
            return np.linalg.solve(A, b)

        # Pre-smoothing
        for _ in range(self.smoother_iterations):
            x = self._gauss_seidel(A, b, x)

        # 残差の計算と制限
        r = b - A @ x
        r_coarse = self._restrict(r)
        A_coarse = self._restrict_matrix(A)
        
        # 粗いグリッドでの補正
        e_coarse = self._v_cycle(A_coarse, r_coarse, 
                                np.zeros_like(r_coarse), level-1)
        
        # 補正の補間と適用
        e = self._prolongate(e_coarse, x.shape)
        x += e
        
        # Post-smoothing
        for _ in range(self.smoother_iterations):
            x = self._gauss_seidel(A, b, x)
            
        return x

    def _gauss_seidel(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Gauss-Seidel平滑化"""
        n = len(b)
        x_new = x.copy()
        
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i,:i], x_new[:i]) 
                       - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
            
        return x_new

    def _restrict(self, fine: np.ndarray) -> np.ndarray:
        """Full-weighting制限演算子"""
        n = len(fine)
        coarse = np.zeros(n//2)
        
        for i in range(n//2):
            if 2*i == 0:
                coarse[i] = (fine[2*i] + fine[2*i+1])/2
            elif 2*i == n-1:
                coarse[i] = (fine[2*i-1] + fine[2*i])/2
            else:
                coarse[i] = (fine[2*i-1] + 2*fine[2*i] + fine[2*i+1])/4
                
        return coarse

    def _restrict_matrix(self, A: np.ndarray) -> np.ndarray:
        """行列の制限"""
        n = len(A)
        A_coarse = np.zeros((n//2, n//2))
        
        for i in range(0, n-1, 2):
            for j in range(0, n-1, 2):
                A_coarse[i//2, j//2] = (A[i,j] + A[i,j+1] + 
                                      A[i+1,j] + A[i+1,j+1])/4
                
        return A_coarse

    def _prolongate(self, coarse: np.ndarray, fine_shape: Tuple[int, ...]) -> np.ndarray:
        """線形補間演算子"""
        fine = np.zeros(fine_shape)
        
        for i in range(len(coarse)):
            fine[2*i] = coarse[i]
            if 2*i+1 < len(fine):
                fine[2*i+1] = (coarse[i] + coarse[i+1])/2 if i < len(coarse)-1 else coarse[i]
                
        return fine