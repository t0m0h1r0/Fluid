import numpy as np
from typing import List, Optional, Tuple
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
            np.full(self.metadata.grid_size, v) 
            for v in self._initial_value
        ]

    @property
    def data(self) -> List[np.ndarray]:
        return self._data

    @data.setter
    def data(self, value: List[np.ndarray]):
        if len(value) != 3:
            raise ValueError("ベクトル場は3成分である必要があります")
        if any(v.shape != self.metadata.grid_size for v in value):
            raise ValueError(f"データのサイズが一致しません")
        self._data = value