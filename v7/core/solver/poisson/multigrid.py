import numpy as np
from typing import Optional, List, Tuple
from core.field.scalar_field import ScalarField
from core.field.metadata import FieldMetadata
from .poisson import PoissonSolver

class MultigridSolver(PoissonSolver):
    """マルチグリッド法"""
    def __init__(self, 
                 num_levels: int = 4,
                 v_cycles: int = 3,
                 pre_smooth: int = 2,
                 post_smooth: int = 2,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_levels = num_levels
        self.v_cycles = v_cycles
        self.pre_smooth = pre_smooth
        self.post_smooth = post_smooth

    def solve(self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None) -> ScalarField:
        if initial_guess is None:
            solution = ScalarField(rhs.metadata)
        else:
            solution = initial_guess

        self.iteration_count = 0
        self.residual_history = []
        
        while self.iteration_count < self.v_cycles:
            solution = self._v_cycle(solution, rhs, self.num_levels)
            
            residual = self.compute_residual(solution, rhs)
            self.residual_history.append(residual)
            
            if self.has_converged():
                break
                
            self.iteration_count += 1
            
        return solution

    def _v_cycle(self, u: ScalarField, f: ScalarField, level: int) -> ScalarField:
        if level == 1:
            return self._solve_directly(f)  # 最粗グリッドでの直接解法

        # Pre-smoothing
        for _ in range(self.pre_smooth):
            u = self._smooth(u, f)

        # 残差の計算
        r = self._compute_residual_field(u, f)

        # 粗いグリッドへの制限
        r_coarse = self._restrict(r)
        metadata_coarse = self._create_coarse_metadata(r.metadata)
        f_coarse = ScalarField(metadata_coarse)
        f_coarse.data = r_coarse

        # 粗いグリッドでの補正（再帰）
        e_coarse = self._v_cycle(ScalarField(metadata_coarse), f_coarse, level-1)

        # 補正の補間と適用
        e = self._prolongate(e_coarse, u.metadata)
        u.data += e.data

        # Post-smoothing
        for _ in range(self.post_smooth):
            u = self._smooth(u, f)

        return u

    def _smooth(self, u: ScalarField, f: ScalarField) -> ScalarField:
        """Gauss-Seidel平滑化"""
        dx2 = [d*d for d in u.dx]
        data = u.data.copy()
        
        for i in range(1, data.shape[0]-1):
            for j in range(1, data.shape[1]-1):
                for k in range(1, data.shape[2]-1):
                    data[i,j,k] = (
                        (data[i+1,j,k] + data[i-1,j,k])/dx2[0] +
                        (data[i,j+1,k] + data[i,j-1,k])/dx2[1] +
                        (data[i,j,k+1] + data[i,j,k-1])/dx2[2] -
                        f.data[i,j,k]
                    ) / (2/dx2[0] + 2/dx2[1] + 2/dx2[2])
        
        u.data = data
        return u

    def _solve_directly(self, f: ScalarField) -> ScalarField:
        """最粗グリッドでの直接解法"""
        # 簡単のためGauss-Seidelを使用
        u = ScalarField(f.metadata)
        for _ in range(50):  # 十分な反復回数
            u = self._smooth(u, f)
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

    def _restrict(self, field: ScalarField) -> np.ndarray:
        """制限演算子 (Full Weighting)"""
        data = field.data
        coarse_shape = tuple(s//2 for s in data.shape)
        coarse_data = np.zeros(coarse_shape)
        
        for i in range(coarse_shape[0]):
            for j in range(coarse_shape[1]):
                for k in range(coarse_shape[2]):
                    i_fine, j_fine, k_fine = 2*i, 2*j, 2*k
                    coarse_data[i,j,k] = np.mean(
                        data[i_fine:i_fine+2,
                             j_fine:j_fine+2,
                             k_fine:k_fine+2]
                    )
        
        return coarse_data

    def _prolongate(self, field: ScalarField, target_metadata: FieldMetadata) -> ScalarField:
        """補間演算子 (Trilinear)"""
        coarse_data = field.data
        fine_shape = target_metadata.resolution
        
        # 補間結果を格納するフィールド
        fine_field = ScalarField(target_metadata)
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

    def _create_coarse_metadata(self, metadata: FieldMetadata) -> FieldMetadata:
        """粗いグリッドのメタデータを作成"""
        return FieldMetadata(
            name=metadata.name,
            unit=metadata.unit,
            domain_size=metadata.domain_size,
            resolution=tuple(s//2 for s in metadata.resolution),
            time=metadata.time
        )