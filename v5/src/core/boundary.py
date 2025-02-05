import numpy as np
from .scheme import BoundaryCondition, StencilOperator

class PeriodicBC(BoundaryCondition):
    def get_stencil_operator(self) -> StencilOperator:
        # 4次精度中心差分の係数
        return StencilOperator(
            points=[-2, -1, 0, 1, 2],
            coefficients=[1/12, -2/3, 0, 2/3, -1/12]
        )
    
    def apply_to_field(self, field: np.ndarray) -> np.ndarray:
        """多次元周期境界条件の実装"""
        # コピーを作成して操作
        periodic_field = field.copy()
        
        # 各次元に対して周期境界を適用
        for axis in range(field.ndim):
            # 端の処理
            slices_start = [slice(None)] * field.ndim
            slices_end = [slice(None)] * field.ndim
            
            # 先頭と末尾のスライスインデックス
            slices_start[axis] = slice(0, 1)
            slices_end[axis] = slice(-1, None)
            
            # 最初の要素を最後の要素でコピー、最後の要素を最初の要素でコピー
            periodic_field[tuple(slices_start)] = field[tuple(slices_end)]
            periodic_field[tuple(slices_end)] = field[tuple(slices_start)]
        
        return periodic_field

class NeumannBC(BoundaryCondition):
    def get_stencil_operator(self) -> StencilOperator:
        # 4次精度片側差分の係数
        return StencilOperator(
            points=[0, 1, 2, 3, 4],
            coefficients=[-25/12, 4, -3, 4/3, -1/4]
        )
    
    def apply_to_field(self, field: np.ndarray) -> np.ndarray:
        """多次元ノイマン境界条件の実装"""
        # コピーを作成して操作
        neumann_field = field.copy()
        
        for axis in range(field.ndim):
            # 各次元の端点を隣接する値に設定
            slices_min = [slice(None)] * field.ndim
            slices_max = [slice(None)] * field.ndim
            
            # 最小側と最大側のスライスインデックス
            slices_min[axis] = 0
            slices_max[axis] = -1
            
            # 境界の値を隣接する値に設定
            neumann_field[tuple(slices_min)] = neumann_field[tuple(
                [slice(None) if i != axis else 1 for i in range(field.ndim)]
            )]
            neumann_field[tuple(slices_max)] = neumann_field[tuple(
                [slice(None) if i != axis else -2 for i in range(field.ndim)]
            )]
        
        return neumann_field

class DirectionalBC:
    """方向毎に異なる境界条件を適用"""
    def __init__(self, x_bc: BoundaryCondition, y_bc: BoundaryCondition, z_bc: BoundaryCondition):
        self.conditions = [x_bc, y_bc, z_bc]
    
    def get_condition(self, axis: int) -> BoundaryCondition:
        return self.conditions[axis]