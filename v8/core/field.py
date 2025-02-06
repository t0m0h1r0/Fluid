from typing import Tuple, Optional, List
import numpy as np
from abc import ABC, abstractmethod

class Field(ABC):
    """場の基底クラス"""
    
    def __init__(self, shape: Tuple[int, ...], dx: float = 1.0):
        """
        Args:
            shape: グリッドの形状
            dx: グリッド間隔
        """
        self._data = np.zeros(shape)
        self.dx = dx
        self._time = 0.0
    
    @property
    def data(self) -> np.ndarray:
        """場のデータを取得"""
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        """場のデータを設定"""
        if value.shape != self._data.shape:
            raise ValueError(f"形状が一致しません: {value.shape} != {self._data.shape}")
        self._data = value
        
    @property
    def shape(self) -> Tuple[int, ...]:
        """場の形状を取得"""
        return self._data.shape
    
    @property
    def ndim(self) -> int:
        """次元数を取得"""
        return self._data.ndim
    
    @property
    def time(self) -> float:
        """現在の時刻を取得"""
        return self._time
    
    @time.setter
    def time(self, value: float):
        """時刻を設定"""
        if value < 0:
            raise ValueError("時刻は非負である必要があります")
        self._time = value

    def gradient(self, axis: int) -> np.ndarray:
        """指定軸方向の勾配を計算（境界での正しい処理を含む）"""
        grad = np.zeros_like(self._data)
        n = self._data.shape[axis]
        
        # Create slices for array indexing
        slice_prev = [slice(None)] * self.ndim
        slice_next = [slice(None)] * self.ndim
        slice_current = [slice(None)] * self.ndim
        
        try:
            # Interior points (central difference)
            slice_current[axis] = slice(1, -1)
            slice_next[axis] = slice(2, None)
            slice_prev[axis] = slice(0, -2)
            
            grad[tuple(slice_current)] = (
                self._data[tuple(slice_next)] - 
                self._data[tuple(slice_prev)]
            ) / (2.0 * self.dx)
            
            # Left boundary (forward difference)
            slice_current[axis] = 0
            slice_next[axis] = 1
            grad[tuple(slice_current)] = (
                self._data[tuple(slice_next)] - 
                self._data[tuple(slice_current)]
            ) / self.dx
            
            # Right boundary (backward difference)
            slice_current[axis] = -1
            slice_prev[axis] = -2
            grad[tuple(slice_current)] = (
                self._data[tuple(slice_current)] - 
                self._data[tuple(slice_prev)]
            ) / self.dx
            
            return grad
            
        except Exception as e:
            print(f"Error in gradient calculation for axis {axis}:")
            print(f"Input shape: {self._data.shape}")
            print(f"Current operation failed: {str(e)}")
            raise
    
    def laplacian(self) -> np.ndarray:
        """ラプラシアンを計算"""
        lap = np.zeros_like(self._data)
        
        for axis in range(self.ndim):
            # Interior points
            slices = [slice(None)] * self.ndim
            slices[axis] = slice(1, -1)
            
            lap[tuple(slices)] += (
                self._data[tuple(slice(2, None) if i == axis else slice(None) 
                              for i in range(self.ndim))] +
                self._data[tuple(slice(0, -2) if i == axis else slice(None) 
                              for i in range(self.ndim))] -
                2 * self._data[tuple(slices)]
            ) / (self.dx ** 2)
            
            # Boundaries (one-sided differences)
            # Left boundary
            slices[axis] = 0
            lap[tuple(slices)] += (
                self._data[tuple(slice(1, 2) if i == axis else slice(None) 
                              for i in range(self.ndim))] -
                self._data[tuple(slices)]
            ) / self.dx
            
            # Right boundary
            slices[axis] = -1
            lap[tuple(slices)] += (
                self._data[tuple(slice(-2, -1) if i == axis else slice(None) 
                              for i in range(self.ndim))] -
                self._data[tuple(slices)]
            ) / self.dx
        
        return lap

class ConservedField(Field):
    """保存則を持つ場の基底クラス"""
    
    def __init__(self, shape: Tuple[int, ...], dx: float = 1.0):
        super().__init__(shape, dx)
        self._initial_integral = self.integrate()
        
    def integrate(self) -> float:
        """場の積分値を計算"""
        return np.sum(self._data) * self.dx**self.ndim
    
    def check_conservation(self) -> float:
        """保存則の確認"""
        current_integral = self.integrate()
        return abs(current_integral - self._initial_integral) / self._initial_integral

class VectorField:
    """ベクトル場クラス"""
    
    def __init__(self, shape: Tuple[int, ...], dx: float = 1.0):
        """
        Args:
            shape: グリッドの形状
            dx: グリッド間隔
        """
        self._components = []
        for _ in range(len(shape)):
            field = Field(shape, dx)
            field.data = np.zeros(shape)  # 明示的に0で初期化
            self._components.append(field)
        self._dx = dx
        self._shape = shape
    
    @property
    def components(self) -> List[Field]:
        """ベクトル場の各成分を取得"""
        return self._components
    
    @property
    def dx(self) -> float:
        """グリッド間隔を取得"""
        return self._dx
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """場の形状を取得"""
        return self._shape
    
    def divergence(self) -> np.ndarray:
        """発散を計算（境界での正しい処理を含む）"""
        div = np.zeros(self.shape)
        
        for i, component in enumerate(self._components):
            div += component.gradient(i)
        
        return div
    
    def curl(self) -> List[np.ndarray]:
        """回転を計算（3次元の場合のみ）"""
        if len(self._components) != 3:
            raise ValueError("回転は3次元ベクトル場でのみ定義されます")
            
        curl = []
        # ∂w/∂y - ∂v/∂z
        curl.append(
            self._components[2].gradient(1) - 
            self._components[1].gradient(2)
        )
        # ∂u/∂z - ∂w/∂x
        curl.append(
            self._components[0].gradient(2) - 
            self._components[2].gradient(0)
        )
        # ∂v/∂x - ∂u/∂y
        curl.append(
            self._components[1].gradient(0) - 
            self._components[0].gradient(1)
        )
        
        return curl