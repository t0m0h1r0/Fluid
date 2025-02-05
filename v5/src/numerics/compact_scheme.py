# numerics/compact_scheme.py
import numpy as np
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import spsolve
from typing import Tuple
from core.scheme import DifferenceScheme, BoundaryCondition, StencilOperator

class CompactScheme(DifferenceScheme):
    def __init__(self, alpha: float = 0.25, a: float = 1.5, b: float = 0.3):
        """
        4次精度コンパクト差分スキームの初期化
        
        Args:
            alpha: コンパクトスキームのパラメータ (典型的には 1/4)
            a: 中心差分の係数 (典型的には 3/2)
            b: 広いステンシルの係数 (典型的には 3/10)
        """
        self.alpha = alpha
        self.a = a
        self.b = b
    
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
        """コンパクトスキームを適用
        
        Args:
            field: 入力フィールド
            boundary_condition: 境界条件
            
        Returns:
            スキームを適用した結果
        """
        # 行列オペレータの生成
        A, B = self.create_operator(len(field), boundary_condition)
        
        # 右辺の計算
        rhs = B @ field
        
        # 線形システムを解く
        result = spsolve(A, rhs)
        
        # 境界条件の適用
        return boundary_condition.apply_to_field(result)
    
    def _build_lhs_matrix(self, size: int) -> lil_matrix:
        """左辺の行列を構築（αf'_{i-1} + f'_i + αf'_{i+1}）"""
        A = lil_matrix((size, size))
        A.setdiag(np.ones(size))  # 主対角
        A.setdiag(self.alpha * np.ones(size-1), k=1)  # 上対角
        A.setdiag(self.alpha * np.ones(size-1), k=-1)  # 下対角
        return A
    
    def _build_rhs_matrix(self, size: int) -> lil_matrix:
        """右辺の行列を構築（af_{i+1} - af_{i-1} + b(f_{i+2} - f_{i-2})）"""
        dx = 1.0  # 正規化された格子間隔
        
        B = lil_matrix((size, size))
        # 中心差分項
        B.setdiag(self.a/dx * np.ones(size-1), k=1)  # f_{i+1}項
        B.setdiag(-self.a/dx * np.ones(size-1), k=-1)  # f_{i-1}項
        
        # 広いステンシル項
        B.setdiag(self.b/dx * np.ones(size-2), k=2)  # f_{i+2}項
        B.setdiag(-self.b/dx * np.ones(size-2), k=-2)  # f_{i-2}項
        
        return B
    
    def _modify_matrices(self, A: lil_matrix, B: lil_matrix, stencil: StencilOperator):
        """境界条件に基づいて行列を修正"""
        size = A.shape[0]
        points = stencil.points
        coeffs = stencil.coefficients
        
        # 前方境界（i=0,1）での修正
        self._apply_boundary_stencil(A, B, points, coeffs, 0)
        self._apply_boundary_stencil(A, B, points, coeffs, 1)
        
        # 後方境界（i=N-2,N-1）での修正
        self._apply_boundary_stencil(A, B, points, coeffs, size-2)
        self._apply_boundary_stencil(A, B, points, coeffs, size-1)
    
    def _apply_boundary_stencil(self, A: lil_matrix, B: lil_matrix, 
                               points: np.ndarray, coeffs: np.ndarray, idx: int):
        """境界ステンシルの適用"""
        size = A.shape[0]
        # 行をクリア
        A[idx, :] = 0
        B[idx, :] = 0
        
        # 新しいステンシルを適用
        for p, c in zip(points, coeffs):
            if 0 <= idx + p < size:
                A[idx, idx + p] = c