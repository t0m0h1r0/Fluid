from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from ..field.base import Field

class BoundaryCondition(ABC):
    """境界条件の基底クラス"""
    
    @abstractmethod
    def apply(self, field: Field) -> Field:
        """境界条件の適用"""
        pass

    @abstractmethod
    def get_ghost_points(self, field: Field) -> List[np.ndarray]:
        """ゴースト点の値を計算"""
        pass

    @abstractmethod
    def validate(self, field: Field) -> bool:
        """境界条件の妥当性検証"""
        pass

class DirectionalBoundaryCondition:
    """方向ごとの境界条件"""
    
    def __init__(self, conditions: List[Tuple[str, BoundaryCondition]]):
        """
        Args:
            conditions: [(方向, 境界条件), ...]
                方向は 'x+', 'x-', 'y+', 'y-', 'z+', 'z-'
        """
        self.conditions = dict(conditions)

    def apply(self, field: Field) -> Field:
        """全方向の境界条件を適用"""
        result = field
        for direction, condition in self.conditions.items():
            result = condition.apply(result)
        return result