import numpy as np
from typing import List, Tuple
from core.scheme import DifferenceScheme, BoundaryCondition

class PoissonSolver:
    def __init__(self, 
                 scheme: DifferenceScheme,
                 boundary_conditions: List[BoundaryCondition],
                 max_iter: int = 1000,
                 tolerance: float = 1e-6):
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.operators = {}  # キャッシュされた差分演算子
    
    def create_operators(self, shape: Tuple[int, ...]):
        """各方向の差分演算子を生成"""
        for axis, (size, bc) in enumerate(zip(shape, self.boundary_conditions)):
            if (axis, size) not in self.operators:
                self.operators[(axis, size)] = self.scheme.create_operator(size, bc)
    
    def solve(self, rhs: np.ndarray, initial_guess: np.ndarray = None) -> np.ndarray:
        """ポアソン方程式を解く"""
        if initial_guess is None:
            p = np.zeros_like(rhs)
        else:
            p = initial_guess.copy()
        
        self.create_operators(rhs.shape)
        
        for iter in range(self.max_iter):
            max_residual = 0
            
            # 各方向の2階微分を計算
            for axis in range(rhs.ndim):
                bc = self.boundary_conditions[axis]
                A, B = self.operators[(axis, rhs.shape[axis])]
                
                # 現在の方向に沿って各ラインを解く
                for idx in self._get_orthogonal_indices(rhs.shape, axis):
                    line = self._get_line(p, axis, idx)
                    d2_line = self.scheme.apply(line, bc)
                    self._set_line(p, axis, idx, d2_line)
            
            # 残差の計算
            residual = self._compute_residual(p, rhs)
            max_residual = np.max(np.abs(residual))
            
            if iter % 100 == 0:
                print(f"Iteration {iter}, Max Residual: {max_residual:.2e}")
            
            if max_residual < self.tolerance:
                print(f"Converged after {iter} iterations")
                break
        
        return p
    
    def _get_orthogonal_indices(self, shape: Tuple[int, ...], axis: int):
        """指定された軸に直交する全インデックスの組み合わせを生成"""
        ranges = [range(s) for i, s in enumerate(shape) if i != axis]
        return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(len(ranges), -1).T
    
    def _get_line(self, array: np.ndarray, axis: int, idx) -> np.ndarray:
        """指定された軸に沿ってラインを抽出"""
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        return array[tuple(idx_list)]
    
    def _set_line(self, array: np.ndarray, axis: int, idx, values: np.ndarray):
        """指定された軸に沿ってラインを設定"""
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        array[tuple(idx_list)] = values
    
    def _compute_residual(self, p: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """残差を計算"""
        laplacian = np.zeros_like(p)
        for axis in range(p.ndim):
            bc = self.boundary_conditions[axis]
            for idx in self._get_orthogonal_indices(p.shape, axis):
                line = self._get_line(p, axis, idx)
                d2_line = self.scheme.apply(line, bc)
                self._set_line(laplacian, axis, idx, d2_line)
        
        return laplacian - rhs