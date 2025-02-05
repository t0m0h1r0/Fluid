import numpy as np
from typing import Optional
from .base import Field, FieldMetadata

class ScalarField(Field):
    """スカラー場"""
    def __init__(self, metadata: FieldMetadata, initial_value: Optional[float] = 0.0):
        self._data = None
        self._initial_value = initial_value
        super().__init__(metadata)

    def _initialize(self):
        """スカラー場の初期化"""
        self._data = np.full(self.metadata.grid_size, self._initial_value)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        if value.shape != self.metadata.grid_size:
            raise ValueError(f"データのサイズが一致しません: {value.shape} != {self.metadata.grid_size}")
        self._data = value