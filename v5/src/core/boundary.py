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
        if field.ndim == 1:
            field[0] = field[-2]
            field[-1] = field[1]
            return field
        else:
            raise NotImplementedError("Multi-dimensional periodic BC not implemented")

class NeumannBC(BoundaryCondition):
    def get_stencil_operator(self) -> StencilOperator:
        # 4次精度片側差分の係数
        return StencilOperator(
            points=[0, 1, 2, 3, 4],
            coefficients=[-25/12, 4, -3, 4/3, -1/4]
        )
    
    def apply_to_field(self, field: np.ndarray) -> np.ndarray:
        if field.ndim == 1:
            field[0] = field[1]
            field[-1] = field[-2]
            return field
        else:
            raise NotImplementedError("Multi-dimensional Neumann BC not implemented")

class DirectionalBC:
    """方向毎に異なる境界条件を適用"""
    def __init__(self, x_bc: BoundaryCondition, y_bc: BoundaryCondition, z_bc: BoundaryCondition):
        self.conditions = [x_bc, y_bc, z_bc]
    
    def get_condition(self, axis: int) -> BoundaryCondition:
        return self.conditions[axis]