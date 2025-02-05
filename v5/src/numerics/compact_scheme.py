import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve
from typing import Tuple
from core.scheme import DifferenceScheme, BoundaryCondition, StencilOperator

class CompactScheme(DifferenceScheme):
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha  # コンパクトスキームのパラメータ
    
    def create_operator(self, size: int, boundary_condition: BoundaryCondition) -> Tuple[lil_matrix, lil_matrix]:
        """左辺と右辺の行列を生成"""
        # 基本行列の構築
        A = self._build_lhs_matrix(size)
        B = self._build_rhs_matrix(size)
        
        # 境界条件に基づく修正
        stencil = boundary_condition.get_stencil_operator()
        self._modify_matrices(A, B, stencil)
        
        return A.tocsr(), B.tocsr()
    
    def apply(self, field: np.ndarray, boundary_condition: BoundaryCondition) -> np.ndarray:
        """コンパクトスキームを適用"""
        A, B = self.create_operator(len(field), boundary_condition)
        rhs = B @ field
        result = spsolve(A, rhs)
        return boundary_condition.apply_to_field(result)
    
    def _build_lhs_matrix(self, size: int) -> lil_matrix:
        """左辺の行列を構築"""
        A = lil_matrix((size, size))
        A.setdiag(np.ones(size))  # 主対角
        A.setdiag(self.alpha * np.ones(size-1), k=1)  # 上対角
        A.setdiag(self.alpha * np.ones(size-1), k=-1)  # 下対角
        return A
    
    def _build_rhs_matrix(self, size: int) -> lil_matrix:
        """右辺の行列を構築"""
        dx = 1.0  # 正規化された格子間隔
        a = 3.0 / (4.0 * dx)  # 4次精度の係数
        
        B = lil_matrix((size, size))
        B.setdiag(np.zeros(size))  # 主対角
        B.setdiag(a * np.ones(size-1), k=1)  # 上対角
        B.setdiag(-a * np.ones(size-1), k=-1)  # 下対角
        return B
    
    def _modify_matrices(self, A: lil_matrix, B: lil_matrix, stencil: StencilOperator):
        """境界条件に基づいて行列を修正"""
        size = A.shape[0]
        points = stencil.points
        coeffs = stencil.coefficients
        
        # 境界での修正（前方）
        for i, (p, c) in enumerate(zip(points, coeffs)):
            if 0 <= p < size:
                A[0, p] = c
        
        # 境界での修正（後方）
        for i, (p, c) in enumerate(zip(points, coeffs)):
            if 0 <= size-1+p < size:
                A[-1, size-1+p] = c