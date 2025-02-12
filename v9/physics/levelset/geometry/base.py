"""Level Set法での幾何学的計算の基底クラスを提供するモジュール"""

from abc import ABC, abstractmethod
import numpy as np


class BaseGeometryCalculator(ABC):
    """幾何学的計算の基底クラス"""

    def __init__(self, dx: float):
        """
        Args:
            dx: グリッド間隔
        """
        if dx <= 0:
            raise ValueError("dxは正の値である必要があります")
        self.dx = dx

    @abstractmethod
    def compute(self, phi: np.ndarray, **kwargs) -> np.ndarray:
        """幾何学的量を計算

        Args:
            phi: Level Set関数の値
            **kwargs: 追加のパラメータ

        Returns:
            計算された幾何学的量
        """
        pass

    def _compute_gradient(self, phi: np.ndarray) -> np.ndarray:
        """勾配を計算"""
        grad = np.array(np.gradient(phi, self.dx))
        return grad

    def _compute_gradient_norm(self, phi: np.ndarray) -> np.ndarray:
        """勾配の大きさを計算"""
        grad = self._compute_gradient(phi)
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))
        return np.maximum(grad_norm, 1e-10)  # ゼロ除算防止
