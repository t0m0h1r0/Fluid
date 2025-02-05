import numpy as np
from typing import Optional, Tuple
from .base import Field, FieldMetadata

class ScalarField(Field):
    """スカラー場"""
    def __init__(self, metadata: FieldMetadata, initial_value: Optional[float] = 0.0):
        self._data = None
        self._initial_value = initial_value
        super().__init__(metadata)

    def _initialize(self):
        """スカラー場の初期化"""
        # metadata.resolutionをタプルに変換して使用
        resolution = tuple(self.metadata.resolution)
        self._data = np.full(resolution, self._initial_value)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        # metadata.resolutionをタプルに変換して比較
        resolution = tuple(self.metadata.resolution)
        if value.shape != resolution:
            raise ValueError(f"データのサイズが一致しません: {value.shape} != {resolution}")
        self._data = value