"""Level Set法での法線ベクトル計算を提供するモジュール"""

import numpy as np
from core.field import VectorField, ScalarField
from .base import BaseGeometryCalculator


class NormalCalculator(BaseGeometryCalculator):
    """法線ベクトル計算クラス"""

    def compute(self, phi: np.ndarray, **kwargs) -> VectorField:
        """法線ベクトルを計算

        Args:
            phi: Level Set関数の値
            **kwargs: 未使用のキーワード引数

        Returns:
            法線ベクトルを表すVectorField
        """
        # 勾配と勾配の大きさを計算
        grad = self._compute_gradient(phi)
        grad_norm = self._compute_gradient_norm(phi)

        # VectorFieldの作成
        normal = VectorField(phi.shape, self.dx)
        for i, grad_i in enumerate(grad):
            normal.components[i] = ScalarField(
                phi.shape, self.dx, initial_value=grad_i / grad_norm
            )

        return normal
