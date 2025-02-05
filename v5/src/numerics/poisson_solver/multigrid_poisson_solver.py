# numerics/poisson_solver/multigrid_poisson_solver.py
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from .abstract_poisson_solver import AbstractPoissonSolver
from core.boundary import BoundaryCondition

@dataclass
class MultigridConfig:
    """マルチグリッドソルバーの設定"""
    max_levels: int = 4           # マルチグリッドレベルの最大数
    pre_smoothing: int = 2        # 前処理スムージングの回数
    post_smoothing: int = 2       # 後処理スムージングの回数
    coarse_size: int = 4          # 最も粗いグリッドのサイズ
    max_iterations: int = 1000    # 最大反復回数
    tolerance: float = 1e-6       # 収束判定の閾値
    omega: float = 2/3           # SOR法の緩和係数

class MultigridPoissonSolver(AbstractPoissonSolver):
    def __init__(self, 
                 config: Optional[MultigridConfig] = None,
                 boundary_conditions: List[BoundaryCondition] = None):
        """
        マルチグリッドポアソンソルバーの初期化
        
        Args:
            config: ソルバーの設定
            boundary_conditions: 境界条件のリスト [x_bc, y_bc, z_bc]
        """
        self.config = config or MultigridConfig()
        self.boundary_conditions = boundary_conditions
        self.operators = {}  # 各レベルの演算子を保持
    
    def _create_restriction_operator(self, fine_shape: Tuple[int, ...]) -> Callable:
        """制限演算子の生成"""
        def restrict(fine_grid: np.ndarray) -> np.ndarray:
            coarse_shape = tuple(s // 2 for s in fine_grid.shape)
            coarse_grid = np.zeros(coarse_shape)
            
            # 重み付き制限
            weights = np.array([
                [1/16, 1/8, 1/16],
                [1/8,  1/4, 1/8],
                [1/16, 1/8, 1/16]
            ])
            
            for i in range(coarse_shape[0]):
                for j in range(coarse_shape[1]):
                    for k in range(coarse_shape[2]):
                        i_f, j_f, k_f = 2*i, 2*j, 2*k
                        coarse_grid[i,j,k] = np.sum(
                            fine_grid[i_f:i_f+3, j_f:j_f+3, k_f:k_f+3] * 
                            weights[:,:,np.newaxis]
                        )
            
            return coarse_grid
        
        return restrict
    
    def _create_prolongation_operator(self, coarse_shape: Tuple[int, ...]) -> Callable:
        """補間演算子の生成"""
        def prolong(coarse_grid: np.ndarray) -> np.ndarray:
            fine_shape = tuple(2*s for s in coarse_grid.shape)
            fine_grid = np.zeros(fine_shape)
            
            # 三線形補間
            for i in range(coarse_grid.shape[0]):
                for j in range(coarse_grid.shape[1]):
                    for k in range(coarse_grid.shape[2]):
                        i_f, j_f, k_f = 2*i, 2*j, 2*k
                        
                        # 頂点の値
                        fine_grid[i_f, j_f, k_f] = coarse_grid[i,j,k]
                        
                        # エッジの補間
                        if i_f+1 < fine_shape[0]:
                            fine_grid[i_f+1, j_f, k_f] = 0.5 * (
                                coarse_grid[i,j,k] + 
                                coarse_grid[min(i+1,coarse_grid.shape[0]-1),j,k]
                            )
                        if j_f+1 < fine_shape[1]:
                            fine_grid[i_f, j_f+1, k_f] = 0.5 * (
                                coarse_grid[i,j,k] + 
                                coarse_grid[i,min(j+1,coarse_grid.shape[1]-1),k]
                            )
                        if k_f+1 < fine_shape[2]:
                            fine_grid[i_f, j_f, k_f+1] = 0.5 * (
                                coarse_grid[i,j,k] + 
                                coarse_grid[i,j,min(k+1,coarse_grid.shape[2]-1)]
                            )
            
            return fine_grid
        
        return prolong
    
    def _smooth(self, u: np.ndarray, f: np.ndarray, 
               iterations: int, dx: float) -> np.ndarray:
        """ガウス-ザイデル法によるスムージング"""
        omega = self.config.omega
        for _ in range(iterations):
            u_new = u.copy()
            for i in range(1, u.shape[0]-1):
                for j in range(1, u.shape[1]-1):
                    for k in range(1, u.shape[2]-1):
                        u_new[i,j,k] = (1-omega)*u[i,j,k] + omega/6 * (
                            u[i-1,j,k] + u[i+1,j,k] +
                            u[i,j-1,k] + u[i,j+1,k] +
                            u[i,j,k-1] + u[i,j,k+1] -
                            dx**2 * f[i,j,k]
                        )
            u = u_new
            
            # 境界条件の適用
            if self.boundary_conditions:
                for axis, bc in enumerate(self.boundary_conditions):
                    u = bc.apply_to_field(u)
                    
        return u
    
    def _v_cycle(self, u: np.ndarray, f: np.ndarray, 
                 level: int, dx: float) -> np.ndarray:
        """V-サイクルの実行"""
        if level == 0 or min(u.shape) <= self.config.coarse_size:
            # 最も粗いグリッドでは直接解く
            return self._solve_directly(u, f, dx)
            
        # 前処理スムージング
        u = self._smooth(u, f, self.config.pre_smoothing, dx)
        
        # 残差の計算
        r = f - self._apply_laplacian(u, dx)
        
        # 粗いグリッドへの制限
        r_c = self._create_restriction_operator(r.shape)(r)
        e_c = np.zeros_like(r_c)
        
        # 粗いグリッドでの修正
        e_c = self._v_cycle(e_c, r_c, level-1, 2*dx)
        
        # 細かいグリッドへの補間
        e = self._create_prolongation_operator(e_c.shape)(e_c)
        u += e
        
        # 後処理スムージング
        u = self._smooth(u, f, self.config.post_smoothing, dx)
        
        return u
    
    def _solve_directly(self, u: np.ndarray, f: np.ndarray, dx: float) -> np.ndarray:
        """最も粗いグリッドでの直接解法"""
        # ガウス-ザイデル法を十分な回数実行
        return self._smooth(u, f, 50, dx)
    
    def _apply_laplacian(self, u: np.ndarray, dx: float) -> np.ndarray:
        """ラプラシアン演算子の適用"""
        laplacian = np.zeros_like(u)
        for i in range(1, u.shape[0]-1):
            for j in range(1, u.shape[1]-1):
                for k in range(1, u.shape[2]-1):
                    laplacian[i,j,k] = (
                        (u[i+1,j,k] + u[i-1,j,k] +
                         u[i,j+1,k] + u[i,j-1,k] +
                         u[i,j,k+1] + u[i,j,k-1] -
                         6*u[i,j,k]) / dx**2
                    )
        return laplacian
    
    def solve(self, rhs: np.ndarray, 
             initial_guess: Optional[np.ndarray] = None,
             dx: float = 1.0) -> np.ndarray:
        """ポアソン方程式を解く
        
        Args:
            rhs: 右辺
            initial_guess: 初期推定値
            dx: グリッド間隔
            
        Returns:
            解
        """
        if initial_guess is None:
            u = np.zeros_like(rhs)
        else:
            u = initial_guess.copy()
            
        for iter in range(self.config.max_iterations):
            u_old = u.copy()
            
            # V-サイクルの実行
            u = self._v_cycle(u, rhs, self.config.max_levels, dx)
            
            # 収束判定
            residual = np.max(np.abs(u - u_old))
            if residual < self.config.tolerance:
                print(f"  マルチグリッド法: {iter+1}回の反復で収束")
                break
                
        return u