from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Tuple, List
from scipy.sparse.linalg import cg
from core.solver import IterativeSolver
from core.field import Field
from core.boundary import BoundaryCondition

class PoissonSolver(IterativeSolver):
    """ポアソン方程式のソルバーの基底クラス"""
    
    def __init__(self, boundary_condition: BoundaryCondition):
        super().__init__()
        self.boundary_condition = boundary_condition
        
    @abstractmethod
    def solve(self, rhs: np.ndarray, initial_guess: Optional[np.ndarray] = None,
             **kwargs) -> np.ndarray:
        """ポアソン方程式を解く
        
        Args:
            rhs: 右辺
            initial_guess: 初期推定値
            **kwargs: 追加のパラメータ
            
        Returns:
            解
        """
        pass

class JacobiSolver(PoissonSolver):
    """ヤコビ法によるポアソンソルバー"""
    
    def solve(self, rhs: np.ndarray, initial_guess: Optional[np.ndarray] = None,
             **kwargs) -> np.ndarray:
        dx = kwargs.get('dx', 1.0)
        if initial_guess is None:
            initial_guess = np.zeros_like(rhs)
            
        solution = initial_guess.copy()
        dx2 = dx * dx
        
        for _ in range(self.max_iterations):
            old_solution = solution.copy()
            
            # ヤコビ反復
            for axis in range(rhs.ndim):
                solution += (np.roll(old_solution, 1, axis) + 
                           np.roll(old_solution, -1, axis) - 
                           2 * old_solution) / (2 * rhs.ndim)
            solution = dx2 * rhs / (2 * rhs.ndim) + solution
            
            # 境界条件の適用
            solution = self.boundary_condition.apply(solution, axis)
            
            # 収束判定
            if np.max(np.abs(solution - old_solution)) < self.tolerance:
                return solution
                
        raise RuntimeError("ヤコビ法が収束しませんでした")

class SORSolver(PoissonSolver):
    """SOR法によるポアソンソルバー"""
    
    def __init__(self, boundary_condition: BoundaryCondition, omega: float = 1.5):
        super().__init__(boundary_condition)
        self.omega = omega
    
    def solve(self, rhs: np.ndarray, initial_guess: Optional[np.ndarray] = None,
             **kwargs) -> np.ndarray:
        dx = kwargs.get('dx', 1.0)
        if initial_guess is None:
            initial_guess = np.zeros_like(rhs)
            
        solution = initial_guess.copy()
        dx2 = dx * dx
        
        for _ in range(self.max_iterations):
            old_solution = solution.copy()
            
            # SOR反復
            for axis in range(rhs.ndim):
                new_value = (np.roll(solution, 1, axis) + 
                           np.roll(solution, -1, axis) - 
                           2 * solution) / (2 * rhs.ndim)
                new_value = dx2 * rhs / (2 * rhs.ndim) + new_value
                solution += self.omega * (new_value - solution)
            
            # 境界条件の適用
            solution = self.boundary_condition.apply(solution, axis)
            
            # 収束判定
            if np.max(np.abs(solution - old_solution)) < self.tolerance:
                return solution
                
        raise RuntimeError("SOR法が収束しませんでした")

class ConjugateGradientSolver(PoissonSolver):
    """共役勾配法によるポアソンソルバー"""
    
    def solve(self, rhs: np.ndarray, initial_guess: Optional[np.ndarray] = None,
             **kwargs) -> np.ndarray:
        dx = kwargs.get('dx', 1.0)
        if initial_guess is None:
            initial_guess = np.zeros_like(rhs)
        
        # 線形演算子の定義
        def A(x):
            result = np.zeros_like(x)
            for axis in range(x.ndim):
                result += (np.roll(x, 1, axis) +
                         np.roll(x, -1, axis) -
                         2 * x) / dx**2
            return result
        
        # 共役勾配法の実行
        solution, info = cg(A, rhs.flatten(), 
                          x0=initial_guess.flatten(),
                          tol=self.tolerance,
                          maxiter=self.max_iterations)
        
        if info != 0:
            raise RuntimeError("共役勾配法が収束しませんでした")
            
        solution = solution.reshape(rhs.shape)
        
        # 境界条件の適用
        for axis in range(rhs.ndim):
            solution = self.boundary_condition.apply(solution, axis)
            
        return solution

class MultigridSolver(PoissonSolver):
    """マルチグリッド法によるポアソンソルバー"""
    
    def __init__(self, boundary_condition: BoundaryCondition, 
                 num_levels: int = 4,
                 pre_sweeps: int = 2,
                 post_sweeps: int = 2):
        super().__init__(boundary_condition)
        self.num_levels = num_levels
        self.pre_sweeps = pre_sweeps
        self.post_sweeps = post_sweeps
        self.smoother = JacobiSolver(boundary_condition)
        
    def solve(self, rhs: np.ndarray, initial_guess: Optional[np.ndarray] = None,
             **kwargs) -> np.ndarray:
        dx = kwargs.get('dx', 1.0)
        if initial_guess is None:
            initial_guess = np.zeros_like(rhs)
        
        for _ in range(self.max_iterations):
            old_solution = initial_guess.copy()
            
            # マルチグリッドV-サイクル
            initial_guess = self._v_cycle(initial_guess, rhs, dx, self.num_levels)
            
            # 境界条件の適用
            for axis in range(rhs.ndim):
                initial_guess = self.boundary_condition.apply(initial_guess, axis)
            
            # 収束判定
            if np.max(np.abs(initial_guess - old_solution)) < self.tolerance:
                return initial_guess
                
        raise RuntimeError("マルチグリッド法が収束しませんでした")
    
    def _v_cycle(self, u: np.ndarray, f: np.ndarray, dx: float, 
                level: int) -> np.ndarray:
        """マルチグリッドV-サイクルの実行"""
        if level == 1:
            # 最粗グリッドでは直接解く
            return self.smoother.solve(f, u, dx=dx)
        
        # 前平滑化
        for _ in range(self.pre_sweeps):
            u = self.smoother.solve(f, u, dx=dx)
        
        # 残差の計算
        r = f - self._apply_operator(u, dx)
        
        # 制限
        r_coarse = self._restrict(r)
        u_coarse = np.zeros_like(r_coarse)
        
        # 粗グリッドでの補正
        e_coarse = self._v_cycle(u_coarse, r_coarse, 2*dx, level-1)
        
        # 補間
        e = self._prolongate(e_coarse, u.shape)
        
        # 解の修正
        u += e
        
        # 後平滑化
        for _ in range(self.post_sweeps):
            u = self.smoother.solve(f, u, dx=dx)
        
        return u
    
    def _apply_operator(self, u: np.ndarray, dx: float) -> np.ndarray:
        """ラプラス演算子の適用"""
        result = np.zeros_like(u)
        dx2 = dx * dx
        for axis in range(u.ndim):
            result += (np.roll(u, 1, axis) +
                      np.roll(u, -1, axis) -
                      2 * u) / dx2
        return result
    
    def _restrict(self, fine: np.ndarray) -> np.ndarray:
        """制限演算子（Full Weighting）"""
        coarse_shape = tuple(s // 2 for s in fine.shape)
        coarse = np.zeros(coarse_shape)
        
        # 1/4, 1/2, 1/4 の重みによる制限
        for offset in np.ndindex((2,) * fine.ndim):
            weight = 0.5**sum(offset)
            slices = tuple(slice(o, None, 2) for o in offset)
            coarse += weight * fine[slices]
            
        return coarse
    
    def _prolongate(self, coarse: np.ndarray, fine_shape: Tuple[int, ...]) -> np.ndarray:
        """補間演算子（線形補間）"""
        fine = np.zeros(fine_shape)
        
        # まず直接の点を補間
        slices = tuple(slice(None, None, 2) for _ in range(coarse.ndim))
        fine[slices] = coarse
        
        # 各方向の中間点を補間
        for axis in range(coarse.ndim):
            # 前方向の補間
            slices = [slice(None, None, 2)] * coarse.ndim
            slices[axis] = slice(1, None, 2)
            fine[tuple(slices)] = 0.5 * (np.roll(fine[tuple(slices)], 1, axis) +
                                       np.roll(fine[tuple(slices)], -1, axis))
            
        return fine
