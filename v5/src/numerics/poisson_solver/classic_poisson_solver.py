import numpy as np
from typing import List, Optional
from scipy import sparse

from .abstract_poisson_solver import AbstractPoissonSolver
from core.scheme import DifferenceScheme, BoundaryCondition

class ClassicPoissonSolver(AbstractPoissonSolver):
    """
    従来の逐次反復法によるポアソンソルバー
    """
    def solve(self, 
              rhs: np.ndarray, 
              initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ポアソン方程式を解く
        
        Args:
            rhs (np.ndarray): 右辺ベクトル
            initial_guess (Optional[np.ndarray]): 初期解（デフォルトはゼロ）
        
        Returns:
            np.ndarray: ポアソン方程式の解
        """
        # 初期解の設定
        if initial_guess is None:
            solution = np.zeros_like(rhs)
        else:
            solution = initial_guess.copy()
        
        # システム行列の構築（3D対応の簡易実装）
        def create_system_matrix(shape):
            """3D有限差分近似のラプラシアン行列を作成"""
            nx, ny, nz = shape
            N = nx * ny * nz
            
            # スパース行列用のデータ
            row_indices = []
            col_indices = []
            data = []
            
            # 対角要素と近傍要素の重み
            diag_weight = -6.0  # 3D空間での6近傍
            neigh_weight = 1.0
            
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        # 現在の1D インデックス
                        idx = i + j * nx + k * nx * ny
                        
                        # 対角要素
                        row_indices.append(idx)
                        col_indices.append(idx)
                        data.append(diag_weight)
                        
                        # x方向の近傍
                        if i > 0:
                            row_indices.append(idx)
                            col_indices.append(idx - 1)
                            data.append(neigh_weight)
                        if i < nx - 1:
                            row_indices.append(idx)
                            col_indices.append(idx + 1)
                            data.append(neigh_weight)
                        
                        # y方向の近傍
                        if j > 0:
                            row_indices.append(idx)
                            col_indices.append(idx - nx)
                            data.append(neigh_weight)
                        if j < ny - 1:
                            row_indices.append(idx)
                            col_indices.append(idx + nx)
                            data.append(neigh_weight)
                        
                        # z方向の近傍
                        if k > 0:
                            row_indices.append(idx)
                            col_indices.append(idx - nx * ny)
                            data.append(neigh_weight)
                        if k < nz - 1:
                            row_indices.append(idx)
                            col_indices.append(idx + nx * ny)
                            data.append(neigh_weight)
            
            # スパース行列の作成
            return sparse.csr_matrix(
                (data, (row_indices, col_indices)), 
                shape=(N, N)
            )
        
        # ガウス・ザイデル反復法
        def gauss_seidel(A, b, x, num_steps):
            """
            ガウス・ザイデル反復法による平滑化
            
            Args:
                A (sparse.csr_matrix): システム行列
                b (np.ndarray): 右辺ベクトル
                x (np.ndarray): 初期解
                num_steps (int): 反復回数
            
            Returns:
                np.ndarray: 更新された解
            """
            for _ in range(num_steps):
                x_new = x.copy()
                for i in range(len(b)):
                    x_new[i] = (
                        b[i] - 
                        A[i, :i].dot(x_new[:i]) - 
                        A[i, i+1:].dot(x[i+1:])
                    ) / A[i, i]
                x = x_new
            return x
        
        # 主ソルバーループ
        for iteration in range(self.max_iterations):
            # システム行列の作成
            A = create_system_matrix(solution.shape)
            
            # ガウス・ザイデル反復
            solution_old = solution.copy()
            solution = gauss_seidel(
                A, 
                rhs.ravel(), 
                solution.ravel(), 
                num_steps=10
            ).reshape(solution.shape)
            
            # 収束判定
            residual = np.linalg.norm(rhs - (
                A.dot(solution.ravel())
            ).reshape(solution.shape))
            
            # 進捗表示
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Residual: {residual}")
            
            if residual < self.tolerance:
                print(f"Converged after {iteration} iterations")
                break
        
        # 境界条件の適用
        return self._apply_boundary_conditions(solution)
    
    def _apply_boundary_conditions(self, solution: np.ndarray) -> np.ndarray:
        """
        境界条件の適用（各軸の境界条件を個別に処理）
        
        Args:
            solution (np.ndarray): 解
        
        Returns:
            np.ndarray: 境界条件を適用した解
        """
        for axis, bc in enumerate(self.boundary_conditions):
            # 各軸の境界条件を適用
            solution = bc.apply_to_field(solution)
        
        return solution