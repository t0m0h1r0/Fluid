import numpy as np
from typing import List, Optional, Tuple, Union
from .base import Field, FieldMetadata

class VectorField(Field):
    """ベクトル場"""
    def __init__(self, metadata: FieldMetadata, initial_value: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)):
        self._data = None
        self._initial_value = initial_value
        super().__init__(metadata)

    def _initialize(self):
        """ベクトル場の初期化"""
        self._data = [
            np.full(self.metadata.resolution, v) 
            for v in self._initial_value
        ]

    @property
    def data(self) -> List[np.ndarray]:
        return self._data

    @data.setter
    def data(self, value: List[np.ndarray]):
        if len(value) != 3:
            raise ValueError("ベクトル場は3成分である必要があります")
        if any(v.shape != self.metadata.resolution for v in value):
            raise ValueError(f"データのサイズが一致しません")
        self._data = value

    def copy(self) -> 'VectorField':
        """深いコピーを作成"""
        new_field = VectorField(self.metadata)
        new_field.data = [v.copy() for v in self.data]
        return new_field

    def __add__(self, other: Union['VectorField', List[np.ndarray]]) -> 'VectorField':
        """加算演算子"""
        result = self.copy()
        if isinstance(other, VectorField):
            other_data = other.data
        elif isinstance(other, list) and len(other) == 3:
            other_data = other
        else:
            raise TypeError("加算はVectorFieldまたは3次元リストとのみ可能です")
        
        result.data = [a + b for a, b in zip(self.data, other_data)]
        return result

    def __sub__(self, other: Union['VectorField', List[np.ndarray]]) -> 'VectorField':
        """減算演算子"""
        result = self.copy()
        if isinstance(other, VectorField):
            other_data = other.data
        elif isinstance(other, list) and len(other) == 3:
            other_data = other
        else:
            raise TypeError("減算はVectorFieldまたは3次元リストとのみ可能です")
        
        result.data = [a - b for a, b in zip(self.data, other_data)]
        return result

    def __mul__(self, scalar: float) -> 'VectorField':
        """スカラー倍"""
        result = self.copy()
        result.data = [v * scalar for v in self.data]
        return result

    def __rmul__(self, scalar: float) -> 'VectorField':
        """右からのスカラー倍"""
        return self * scalar

    def __truediv__(self, scalar: float) -> 'VectorField':
        """スカラーによる除算"""
        if scalar == 0:
            raise ValueError("ゼロによる除算はできません")
        return self * (1.0 / scalar)

    def dot(self, other: 'VectorField') -> np.ndarray:
        """内積"""
        return sum(a * b for a, b in zip(self.data, other.data))

    def cross(self, other: 'VectorField') -> 'VectorField':
        """外積"""
        result = self.copy()
        a, b = self.data, other.data
        result.data = [
            a[1]*b[2] - a[2]*b[1],  # x component
            a[2]*b[0] - a[0]*b[2],  # y component
            a[0]*b[1] - a[1]*b[0]   # z component
        ]
        return result

    def magnitude(self) -> np.ndarray:
        """ベクトル場の大きさ"""
        return np.sqrt(sum(v*v for v in self.data))