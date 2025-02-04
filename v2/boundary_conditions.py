from enum import Enum, auto
from typing import Callable, Optional
import numpy as np
import numba

class BoundaryType(Enum):
    """境界条件の種類"""
    NEUMANN = auto()
    DIRICHLET = auto()
    PERIODIC = auto()

class BoundaryConditionHandler:
    """境界条件の管理と適用"""
    @staticmethod
    @numba.njit
    def apply_boundary_condition(
        field: np.ndarray, 
        bc_type: int,  # Neumannなど
        axis: int,
        value: Optional[float] = None
    ) -> np.ndarray:
        """境界条件の適用"""
        result = field.copy()
        dims = field.ndim
        
        # 軸に応じたスライシング
        if dims == 3:
            if axis == 0:
                if bc_type == BoundaryType.NEUMANN.value:
                    result[0] = result[1]
                    result[-1] = result[-2]
                elif bc_type == BoundaryType.DIRICHLET.value:
                    result[0] = value if value is not None else 0
                    result[-1] = value if value is not None else 0
                elif bc_type == BoundaryType.PERIODIC.value:
                    result[0] = result[-2]
                    result[-1] = result[1]
            elif axis == 1:
                if bc_type == BoundaryType.NEUMANN.value:
                    result[:, 0] = result[:, 1]
                    result[:, -1] = result[:, -2]
                elif bc_type == BoundaryType.DIRICHLET.value:
                    result[:, 0] = value if value is not None else 0
                    result[:, -1] = value if value is not None else 0
                elif bc_type == BoundaryType.PERIODIC.value:
                    result[:, 0] = result[:, -2]
                    result[:, -1] = result[:, 1]
            else:
                if bc_type == BoundaryType.NEUMANN.value:
                    result[:, :, 0] = result[:, :, 1]
                    result[:, :, -1] = result[:, :, -2]
                elif bc_type == BoundaryType.DIRICHLET.value:
                    result[:, :, 0] = value if value is not None else 0
                    result[:, :, -1] = value if value is not None else 0
                elif bc_type == BoundaryType.PERIODIC.value:
                    result[:, :, 0] = result[:, :, -2]
                    result[:, :, -1] = result[:, :, 1]
        
        return result

    def __init__(self, grid_shape):
        """境界条件の初期設定"""
        self.grid_shape = grid_shape
        self.boundary_conditions = {
            'x_min': BoundaryType.NEUMANN,
            'x_max': BoundaryType.NEUMANN,
            'y_min': BoundaryType.NEUMANN,
            'y_max': BoundaryType.NEUMANN,
            'z_min': BoundaryType.NEUMANN,
            'z_max': BoundaryType.NEUMANN
        }

    def set_boundary_condition(
        self, 
        axis: str, 
        bc_type: BoundaryType, 
        value: Optional[float] = None
    ):
        """境界条件の設定"""
        if axis not in self.boundary_conditions:
            raise ValueError(f"Invalid axis: {axis}")
        self.boundary_conditions[axis] = bc_type

    def apply_conditions(self, field: np.ndarray) -> np.ndarray:
        """全境界条件の適用"""
        axes = [0, 1, 2]
        for axis in axes:
            bc_min = self.boundary_conditions[f'{["x","y","z"][axis]}_min'].value
            bc_max = self.boundary_conditions[f'{["x","y","z"][axis]}_max'].value
            
            field = self.apply_boundary_condition(field, bc_min, axis)
            field = self.apply_boundary_condition(field, bc_max, axis)
        
        return field

class CustomBoundaryCondition:
    """カスタム境界条件の実装例"""
    @staticmethod
    def slip_wall(field: np.ndarray, axis: int) -> np.ndarray:
        """滑り壁境界条件"""
        result = field.copy()
        if axis == 0:
            result[0] = 0
            result[-1] = 0
        elif axis == 1:
            result[:, 0] = 0
            result[:, -1] = 0
        else:
            result[:, :, 0] = 0
            result[:, :, -1] = 0
        return result
