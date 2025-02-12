"""Level Set操作の基底クラスを提供するモジュール"""

from abc import ABC, abstractmethod
import numpy as np


class BaseLevelSetOperation(ABC):
    """Level Set操作の基底クラス"""

    def __init__(self, dx: float):
        """
        Args:
            dx: グリッド間隔
        """
        if dx <= 0:
            raise ValueError("dxは正の値である必要があります")
        self.dx = dx

    @abstractmethod
    def validate_input(self, phi: np.ndarray) -> None:
        """入力データを検証"""
        pass
