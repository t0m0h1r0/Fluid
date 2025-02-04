from abc import ABC, abstractmethod
import numpy as np

class BoundaryCondition(ABC):
    """境界条件の基底クラス"""
    @abstractmethod
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        """境界条件を適用"""
        pass
    
    def apply_to_vector_field(self, vector_field):
        """ベクトル場全体に境界条件を適用"""
        vector_field.apply_bc(self)

class NeumannBC(BoundaryCondition):
    """ノイマン境界条件"""
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        f = field.copy()
        if axis == 0:
            f[0] = f[1]
            f[-1] = f[-2]
        elif axis == 1:
            f[:, 0] = f[:, 1]
            f[:, -1] = f[:, -2]
        else:
            f[:, :, 0] = f[:, :, 1]
            f[:, :, -1] = f[:, :, -2]
        return f

class DirichletBC(BoundaryCondition):
    """ディリクレ境界条件"""
    def __init__(self, value: float = 0.0):
        self.value = value
        
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        f = field.copy()
        if axis == 0:
            f[0] = self.value
            f[-1] = self.value
        elif axis == 1:
            f[:, 0] = self.value
            f[:, -1] = self.value
        else:
            f[:, :, 0] = self.value
            f[:, :, -1] = self.value
        return f

class PeriodicBC(BoundaryCondition):
    """周期境界条件"""
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        f = field.copy()
        if axis == 0:
            f[0] = f[-2]
            f[-1] = f[1]
        elif axis == 1:
            f[:, 0] = f[:, -2]
            f[:, -1] = f[:, 1]
        else:
            f[:, :, 0] = f[:, :, -2]
            f[:, :, -1] = f[:, :, 1]
        return f
