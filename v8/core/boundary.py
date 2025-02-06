from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class StencilCoefficients:
    """差分ステンシルの係数"""
    points: np.ndarray  # 格子点のインデックス
    coefficients: np.ndarray  # 対応する係数

class BoundaryCondition(ABC):
    """境界条件の基底クラス"""
    
    @abstractmethod
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        """境界条件を適用"""
        pass
    
    @abstractmethod
    def get_stencil(self) -> StencilCoefficients:
        """差分ステンシルの係数を取得"""
        pass

class PeriodicBoundary(BoundaryCondition):
    """周期境界条件"""
    
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        """周期境界条件を適用"""
        result = field.copy()
        
        # 2点のステンシルを使用
        for i in range(2):
            # 前方境界
            slices = [slice(None)] * field.ndim
            slices[axis] = i
            opposite = slices.copy()
            opposite[axis] = -2 + i
            result[tuple(slices)] = field[tuple(opposite)]
            
            # 後方境界
            slices[axis] = -(i + 1)
            opposite[axis] = 1 - i
            result[tuple(slices)] = field[tuple(opposite)]
            
        return result
    
    def get_stencil(self) -> StencilCoefficients:
        """4次精度中心差分の係数を返す"""
        return StencilCoefficients(
            points=np.array([-2, -1, 0, 1, 2]),
            coefficients=np.array([1/12, -2/3, 0, 2/3, -1/12])
        )

class NeumannBoundary(BoundaryCondition):
    """ノイマン境界条件"""
    
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        """ノイマン境界条件を適用"""
        result = field.copy()
        dx = 1.0  # 正規化された格子間隔
        
        # 2点のステンシルを使用
        for i in range(2):
            # 前方境界
            slices = [slice(None)] * field.ndim
            slices[axis] = i
            next_slice = slices.copy()
            next_slice[axis] = i + 1
            result[tuple(slices)] = (
                result[tuple(next_slice)] - self.value * dx
            )
            
            # 後方境界
            slices[axis] = -(i + 1)
            prev_slice = slices.copy()
            prev_slice[axis] = -(i + 2)
            result[tuple(slices)] = (
                result[tuple(prev_slice)] + self.value * dx
            )
            
        return result
    
    def get_stencil(self) -> StencilCoefficients:
        """4次精度片側差分の係数を返す"""
        return StencilCoefficients(
            points=np.array([0, 1, 2, 3, 4]),
            coefficients=np.array([-25/12, 4, -3, 4/3, -1/4])
        )

class DirichletBoundary(BoundaryCondition):
    """ディリクレ境界条件"""
    
    def __init__(self, value: float):
        self.value = value
    
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        """ディリクレ境界条件を適用"""
        result = field.copy()
        
        # 境界値を設定
        slices = [slice(None)] * field.ndim
        
        # 前方境界
        slices[axis] = 0
        result[tuple(slices)] = self.value
        
        # 後方境界
        slices[axis] = -1
        result[tuple(slices)] = self.value
        
        return result
    
    def get_stencil(self) -> StencilCoefficients:
        """4次精度片側差分の係数を返す"""
        return StencilCoefficients(
            points=np.array([0, 1, 2, 3, 4]),
            coefficients=np.array([1, -4, 6, -4, 1])
        )

class DirectionalBoundary:
    """方向ごとの境界条件を管理"""
    
    def __init__(self, conditions: List[BoundaryCondition]):
        """
        Args:
            conditions: 各方向の境界条件のリスト
        """
        self.conditions = conditions
        
    def apply_all(self, field: np.ndarray) -> np.ndarray:
        """全方向に境界条件を適用"""
        result = field.copy()
        for axis, bc in enumerate(self.conditions):
            result = bc.apply(result, axis)
        return result
    
    def get_condition(self, axis: int) -> BoundaryCondition:
        """指定された方向の境界条件を取得"""
        if not 0 <= axis < len(self.conditions):
            raise ValueError(f"無効な軸: {axis}")
        return self.conditions[axis]

class MixedBoundary(BoundaryCondition):
    """混合境界条件"""
    
    def __init__(self, 
                 upper: BoundaryCondition, 
                 lower: BoundaryCondition):
        self.upper = upper
        self.lower = lower
    
    def apply(self, field: np.ndarray, axis: int) -> np.ndarray:
        """混合境界条件を適用"""
        result = field.copy()
        
        # 上側境界に上側の境界条件を適用
        slices_upper = [slice(None)] * field.ndim
        slices_upper[axis] = slice(-2, None)
        temp = np.take(result, indices=range(-2, 0), axis=axis)
        temp = self.upper.apply(temp, axis)
        np.put_along_axis(result, indices=range(-2, 0), values=temp, axis=axis)
        
        # 下側境界に下側の境界条件を適用
        slices_lower = [slice(None)] * field.ndim
        slices_lower[axis] = slice(0, 2)
        temp = np.take(result, indices=range(0, 2), axis=axis)
        temp = self.lower.apply(temp, axis)
        np.put_along_axis(result, indices=range(0, 2), values=temp, axis=axis)
        
        return result
    
    def get_stencil(self) -> StencilCoefficients:
        """上側境界のステンシルを返す"""
        return self.upper.get_stencil()
