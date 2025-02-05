import numpy as np
from typing import Optional, List, Tuple
from .base import PoissonSolver
from .jacobi import JacobiSolver
from ...field.scalar_field import ScalarField

class MultigridSolver(PoissonSolver):
    """マルチグリッド法によるPoisson方程式ソルバー"""
    
    def __init__(self, 
                 num_levels: int = 4,
                 num_vcycles: int = 10,
                 pre_smooth: int = 2,
                 post_smooth: int = 2,
                 **kwargs):
        """
        Args:
            num_levels: グリッドレベル数
            num_vcycles: Vサイクル数
            pre_smooth: 制限前の平滑化回数
            post_smooth: 補間後の平滑化回数
        """
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.num_vcycles = num_vcycles
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth
        self.smoother = JacobiSolver(omega=0.8, max_iterations=1)

    def solve(self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None) -> ScalarField:
        if initial_guess is None:
            solution = ScalarField(rhs.metadata)
        else:
            solution = initial_guess

        self.iteration_count = 0
        self.residual_history = []

        # Vサイクルの実行
        for cycle in range(self.num_vcycles):
            solution = self._v_cycle(solution, rhs, self.num_levels)
            
            # 収束判定
            residual = self.compute_residual(solution, rhs)
            self.residual_history.append(residual)
            
            if self.has_converged():
                break
                
            self.iteration_count += 1

        return solution

    def _v_cycle(self, u: ScalarField, f: ScalarField, level: int) -> ScalarField:
        if level == 1:
            # 最粗グリッドでの直接解法
            return self.smoother.solve(f, u)

        # 事前平滑化
        for _ in range(self.pre_smooth):
            u = self.smoother.solve(f, u)

        # 残差の計算
        residual = self._compute_residual_field(u, f)

        # 制限
        r_coarse = self._restrict(residual)
        u_coarse = self._restrict(u)
        
        # 粗いグリッドでの解法（再帰）
        e_coarse = self._v_cycle(u_coarse, r_coarse, level-1)

        # 補間と補正
        e_fine = self._prolongate(e_coarse)
        u.data += e_fine.data

        # 事後平滑化
        for _ in range(self.post_smooth):
            u = self.smoother.solve(f, u)

        return u

    def _compute_residual_field(self, u: ScalarField, f: ScalarField) -> ScalarField:
        """残差場の計算"""
        residual = ScalarField(u.metadata)
        dx = u.dx
        
        laplacian = np.zeros_like(u.data)
        for i in range(3):
            laplacian += np.gradient(
                np.gradient(u.data, dx[i], axis=i),
                dx[i], axis=i
            )
        
        residual.data = f.data - laplacian
        return residual

    def _restrict(self, field: ScalarField) -> ScalarField:
        """制限演算子 (Full Weighting)"""
        fine_data = field.data
        coarse_shape = tuple(s//2 for s in fine_data.shape)
        
        # メタデータの更新
        coarse_metadata = field.metadata
        coarse_metadata.grid_size = coarse_shape
        coarse_field = ScalarField(coarse_metadata)
        
        # 制限の実行
        coarse_data = np.zeros(coarse_shape)
        for i in range(coarse_shape[0]):
            for j in range(coarse_shape[1]):
                for k in range(coarse_shape[2]):
                    i_fine, j_fine, k_fine = 2*i, 2*j, 2*k
                    coarse_data[i,j,k] = (
                        fine_data[i_fine:i_fine+2,
                                j_fine:j_fine+2,
                                k_fine:k_fine+2]
                    ).mean()
        
        coarse_field.data = coarse_data
        return coarse_field

    def _prolongate(self, field: ScalarField) -> ScalarField:
        """補間演算子 (Trilinear)"""
        coarse_data = field.data
        fine_shape = tuple(2*s for s in coarse_data.shape)
        
        # メタデータの更新
        fine_metadata = field.metadata
        fine_metadata.grid_size = fine_shape
        fine_field = ScalarField(fine_metadata)
        
        # 補間の実行
        fine_data = np.zeros(fine_shape)
        
        # 頂点の値をコピー
        for i in range(coarse_data.shape[0]):
            for j in range(coarse_data.shape[1]):
                for k in range(coarse_data.shape[2]):
                    fine_data[2*i,2*j,2*k] = coarse_data[i,j,k]
        
        # エッジの補間
        for i in range(fine_shape[0]-1):
            for j in range(fine_shape[1]-1):
                for k in range(fine_shape[2]-1):
                    if i%2 == 1:  # x方向の補間
                        fine_data[i,j,k] = 0.5 * (
                            fine_data[i-1,j,k] + fine_data[i+1,j,k]
                        )
                    if j%2 == 1:  # y方向の補間
                        fine_data[i,j,k] = 0.5 * (
                            fine_data[i,j-1,k] + fine_data[i,j+1,k]
                        )
                    if k%2 == 1:  # z方向の補間
                        fine_data[i,j,k] = 0.5 * (
                            fine_data[i,j,k-1] + fine_data[i,j,k+1]
                        )
        
        fine_field.data = fine_data
        return fine_field