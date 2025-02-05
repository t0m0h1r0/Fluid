from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from .metadata import FieldMetadata

class Field(ABC):
    """物理場の基底クラス"""
    def __init__(self, metadata: FieldMetadata):
        self.metadata = metadata
        self._validate_metadata()
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """場の初期化"""
        pass

    def _validate_metadata(self):
        """メタデータの検証"""
        if len(self.metadata.domain_size) != 3 or len(self.metadata.resolution) != 3:
            raise ValueError("domain_size と resolution は3次元である必要があります")
        if any(s <= 0 for s in self.metadata.domain_size + self.metadata.resolution):
            raise ValueError("domain_size と resolution は正の値である必要があります")

    @property
    def dx(self) -> Tuple[float, float, float]:
        """格子間隔"""
        return tuple(d/n for d, n in zip(self.metadata.domain_size, self.metadata.resolution))

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """場のデータ"""
        pass