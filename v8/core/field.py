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
        """指定軸方向の勾配を計算"""
        return np.gradient(self._data, self.dx, axis=axis)
    
    def laplacian(self) -> np.ndarray:
        """ラプラシアンを計算"""
        return sum(np.gradient(np.gradient(self._data, self.dx, axis=i), 
                             self.dx, axis=i) for i in range(self.ndim))

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
        self._components = [Field(shape, dx) for _ in range(len(shape))]
        self._dx = dx
    
    @property
    def components(self) -> List[Field]:
        """ベクトル場の各成分を取得"""
        return self._components
    
    @property
    def dx(self) -> float:
        """グリッド間隔を取得"""
        return self._dx
    
    def divergence(self) -> np.ndarray:
        """発散を計算"""
        div = np.zeros_like(self._components[0].data)
        for i, component in enumerate(self._components):
            div += component.gradient(i)
        return div
    
    def curl(self) -> List[np.ndarray]:
        """回転を計算（3次元の場合のみ）"""
        if len(self._components) != 3:
            raise ValueError("回転は3次元ベクトル場でのみ定義されます")
            
        curl = []
        u, v, w = [c.data for c in self._components]
        dx = self._dx
        
        # ∂w/∂y - ∂v/∂z
        curl.append(np.gradient(w, dx, axis=1) - np.gradient(v, dx, axis=2))
        # ∂u/∂z - ∂w/∂x
        curl.append(np.gradient(u, dx, axis=2) - np.gradient(w, dx, axis=0))
        # ∂v/∂x - ∂u/∂y
        curl.append(np.gradient(v, dx, axis=0) - np.gradient(u, dx, axis=1))
        
        return curl