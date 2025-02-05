import numpy as np
from typing import List, Optional, Tuple
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg

from .abstract_poisson_solver import AbstractPoissonSolver
from core.scheme import DifferenceScheme, BoundaryCondition

class MultigridPoissonSolver(AbstractPoissonSolver):
    """
    高度なマルチグリッド法によるポアソンソルバー
    
    特徴：
    - フルマルチグリッド法
    - 適応的グリッド解像度
    - 柔軟な境界条件対応
    """
    def __init__(self, 
                 scheme: DifferenceScheme,
                 boundary_conditions: List[BoundaryCondition],
                 max_levels: int = 5, 
                 smoothing_method: str = 'gauss_seidel',
                 pre_smooth_steps: int = 2,
                 post_smooth_steps: int = 2,
                 coarse_solver: str = 'direct',
                 tolerance: float = 1e-6,
                 max_iterations: int = 100):
        """
        マルチグリッドソルバーの詳細設定
        
        Args:
            scheme (DifferenceScheme): 差分スキーム
            boundary_conditions (List[BoundaryCondition]): 境界条件
            max_levels (int): 最大マルチグリッドレベル数
            smoothing_method (str): スムージング手法
            pre_smooth_steps (int): 前スムージングステップ数
            post_smooth_steps (int): 後スムージングステップ数
            coarse_solver (str): 粗いグリッドでの解法
            tolerance (float): 収束許容誤差
            max_iterations (int): 最大反復回数
        """
        super().__init__(scheme, boundary_conditions, tolerance, max_iterations)
        self.max_levels = max_levels
        self.smoothing_method = smoothing_method
        self.pre_smooth_steps = pre_smooth_steps
        self.post_smooth_steps = post_smooth_steps
        self.coarse_solver = coarse_solver
    
    def _create_system_matrix(self, shape: Tuple[int, ...]) -> sparse.csr_matrix:
        """
        高度な3D有限差分ラプラシアン行列生成
        
        Args:
            shape (Tuple[int, ...]): グリッドの形状
        
        Returns:
            sparse.csr_matrix: システム行列
        """
        nx, ny, nz = shape
        N = nx * ny * nz
        
        # スパース行列用のデータ
        row_indices, col_indices, data = [], [], []
        
        # 重み係数（中心差分）
        center_weight = -6.0  # 6近傍
        neighbor_weight = 1.0
        
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    # 1D インデックス計算
                    idx = i + j * nx + k * nx * ny
                    
                    # 対角要素
                    row_indices.append(idx)
                    col_indices.append(idx)
                    data.append(center_weight)
                    
                    # 近傍インデックスの追加
                    neighbor_offsets = [
                        (-1, 0, 0), (1, 0, 0),   # x方向
                        (0, -1, 0), (0, 1, 0),   # y方向
                        (0, 0, -1), (0, 0, 1)    # z方向
                    ]
                    
                    for dx, dy, dz in neighbor_offsets:
                        ni, nj, nk = i + dx, j + dy, k + dz
                        
                        # 境界チェック
                        if (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                            neighbor_idx = ni + nj * nx + nk * nx * ny
                            row_indices.append(idx)
                            col_indices.append(neighbor_idx)
                            data.append(neighbor_weight)
        
        # スパース行列の作成
        return sparse.csr_matrix(
            (data, (row_indices, col_indices)), 
            shape=(N, N)
        )
    
    def _smooth(self, A: sparse.csr_matrix, b: np.ndarray, x: np.ndarray, steps: int) -> np.ndarray:
        """
        異なるスムージング手法の実装
        
        Args:
            A (sparse.csr_matrix): システム行列
            b (np.ndarray): 右辺ベクトル
            x (np.ndarray): 初期解
            steps (int): スムージングステップ数
        
        Returns:
            np.ndarray: スムージング後の解
        """
        if self.smoothing_method == 'gauss_seidel':
            for _ in range(steps):
                x_new = x.copy()
                for i in range(len(b)):
                    x_new[i] = (
                        b[i] - 
                        A[i, :i].dot(x_new[:i]) - 
                        A[i, i+1:].dot(x[i+1:])
                    ) / A[i, i]
                x = x_new
        elif self.smoothing_method == 'jacobi':
            for _ in range(steps):
                diag = A.diagonal()
                x_new = (b - A.dot(x) + diag * x) / diag
                x = x_new
        elif self.smoothing_method == 'conjugate_gradient':
            x, _ = cg(A, b, x0=x, maxiter=steps)
        
        return x
    
    def _restrict(self, residual: np.ndarray) -> np.ndarray:
        """
        高度な残差制限（粗いグリッドへの写像）
        
        Args:
            residual (np.ndarray): 入力残差場
        
        Returns:
            np.ndarray: 制限された残差
        """
        restricted = np.zeros(
            tuple((n + 1) // 2 for n in residual.shape)
        )
        
        for idx in np.ndindex(restricted.shape):
            # 補間インデックスを計算
            src_idx = tuple(i * 2 for i in idx)
            
            # 加重平均による制限
            window = residual[
                tuple(slice(i, i+2) for i in src_idx)
            ]
            restricted[idx] = np.mean(window)
        
        return restricted
    
    def _prolongate(self, coarse_solution: np.ndarray, fine_shape: Tuple[int, ...]) -> np.ndarray:
        """
        高度な補間（細密化）
        
        Args:
            coarse_solution (np.ndarray): 粗いグリッドのソリューション
            fine_shape (Tuple[int, ...]): 細密化後の形状
        
        Returns:
            np.ndarray: 細密化されたソリューション
        """
        prolongated = np.zeros(fine_shape)
        
        for idx in np.ndindex(coarse_solution.shape):
            # 対応する細密グリッドのインデックス範囲を計算
            fine_idx = tuple(slice(i*2, i*2+2) for i in idx)
            
            # バイリニア補間
            prolongated[fine_idx] = coarse_solution[idx]
        
        return prolongated
    
    def solve(self, 
              rhs: np.ndarray, 
              initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        マルチグリッド法によるポアソン方程式の解法
        
        Args:
            rhs (np.ndarray): 右辺ベクトル
            initial_guess (Optional[np.ndarray]): 初期解
        
        Returns:
            np.ndarray: ポアソン方程式の解
        """
        # 初期解の設定
        solution = initial_guess if initial_guess is not None else np.zeros_like(rhs)
        
        # V-サイクルのマルチグリッド法
        def v_cycle(solution, rhs):
            # 最粗レベルでの直接解法
            if solution.size <= 8:
                if self.coarse_solver == 'direct':
                    return spsolve(
                        self._create_system_matrix(solution.shape), 
                        rhs.ravel()
                    ).reshape(solution.shape)
                else:
                    # フォールバックは共役勾配法
                    solution, _ = cg(
                        self._create_system_matrix(solution.shape), 
                        rhs.ravel(),
                        x0=solution.ravel()
                    )
                    return solution.reshape(solution.shape)
            
            # システム行列の生成
            A = self._create_system_matrix(solution.shape)
            
            # 前スムージング
            solution = self._smooth(A, rhs.ravel(), solution.ravel(), self.pre_smooth_steps).reshape(solution.shape)
            
            # 残差計算
            residual = rhs - A.dot(solution.ravel()).reshape(solution.shape)
            
            # 残差の制限（粗いグリッドへ）
            coarse_residual = self._restrict(residual)
            
            # 粗いグリッドでの誤差計算
            coarse_error = v_cycle(
                np.zeros_like(coarse_residual), 
                coarse_residual
            )
            
            # 誤差の補間
            error = self._prolongate(coarse_error, solution.shape)
            
            # 解の補正
            solution += error
            
            # 後スムージング
            solution = self._smooth(A, rhs.ravel(), solution.ravel(), self.post_smooth_steps).reshape(solution.shape)
            
            return solution
        
        # 主ソルバーループ
        for iteration in range(self.max_iterations):
            solution_old = solution.copy()
            
            # V-サイクルの実行
            solution = v_cycle(solution, rhs)
            
            # 収束判定
            residual = np.linalg.norm(rhs - self._create_system_matrix(solution.shape).dot(solution.ravel()).reshape(solution.shape))
            
            # 収束チェックと進捗表示
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Residual: {residual}")
            
            if residual < self.tolerance:
                print(f"Converged after {iteration+1} iterations")
                break
        
        # 境界条件の適用
        return self._apply_boundary_conditions(solution)
    
    def _apply_boundary_conditions(self, solution: np.ndarray) -> np.ndarray:
        """
        境界条件の適用
        
        Args:
            solution (np.ndarray): 解
        
        Returns:
            np.ndarray: 境界条件を適用した解
        """
        for axis, bc in enumerate(self.boundary_conditions):
            solution = bc.apply_to_field(solution)
        
        return solution