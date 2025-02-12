"""Level Set法での曲率計算を提供するモジュール"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import BaseGeometryCalculator


class CurvatureCalculator(BaseGeometryCalculator):
    """曲率計算クラス"""

    def compute(self, phi: np.ndarray, method: str = "standard") -> np.ndarray:
        """曲率を計算

        Args:
            phi: Level Set関数の値
            method: 計算手法 ('standard' または 'high_order')

        Returns:
            計算された曲率
        """
        if method == "standard":
            return self._compute_standard_curvature(phi)
        elif method == "high_order":
            return self._compute_high_order_curvature(phi)
        else:
            raise ValueError(f"未知の計算手法: {method}")

    def _compute_standard_curvature(self, phi: np.ndarray) -> np.ndarray:
        """標準的な曲率計算

        Args:
            phi: Level Set関数の値

        Returns:
            計算された曲率
        """
        # 勾配と勾配の大きさを計算
        grad = self._compute_gradient(phi)
        grad_norm = self._compute_gradient_norm(phi)

        # 曲率の計算
        kappa = np.zeros_like(phi)
        for i in range(len(grad)):
            kappa += np.gradient(grad[i] / grad_norm, self.dx, axis=i)

        # オプションで平滑化
        kappa = gaussian_filter(kappa, sigma=self.dx)

        return kappa

    def _compute_high_order_curvature(self, phi: np.ndarray) -> np.ndarray:
        """高次精度の曲率計算

        4次精度の差分スキームを使用して曲率を計算します。
        これにより、より正確な界面の曲率が得られますが、
        計算コストは標準的な方法よりも高くなります。

        Args:
            phi: Level Set関数の値

        Returns:
            計算された曲率
        """
        # 勾配と勾配の大きさを計算
        grad = self._compute_gradient(phi)
        grad_norm = self._compute_gradient_norm(phi)

        # 4次精度の差分で曲率を計算
        kappa = np.zeros_like(phi)
        ndim = len(grad)

        for i in range(ndim):
            for j in range(ndim):
                # 2階微分を4次精度で計算
                if i == j:
                    # 対角項の4次精度の差分
                    # f''''(x) ≈ (-f[x+2] + 16f[x+1] - 30f[x] + 16f[x-1] - f[x-2]) / (12dx²)
                    d2 = (
                        -phi[..., :-4]
                        + 16 * phi[..., 1:-3]
                        - 30 * phi[..., 2:-2]
                        + 16 * phi[..., 3:-1]
                        - phi[..., 4:]
                    ) / (12 * self.dx**2)
                else:
                    # 交差微分項の4次精度の差分
                    # f''_xy(x,y) ≈ (f[x+1,y+1] - f[x+1,y-1] - f[x-1,y+1] + f[x-1,y-1]) / (4dx²)
                    d2 = (
                        phi[..., 2:, 2:]
                        - phi[..., :-2, 2:]
                        - phi[..., 2:, :-2]
                        + phi[..., :-2, :-2]
                    ) / (4 * self.dx**2)

                kappa += d2 / grad_norm

        # パディングの処理
        # 高次差分により失われた境界部分を補完
        pad_width = 2
        kappa = np.pad(kappa, pad_width, mode="edge")

        return kappa

    def validate_input(self, phi: np.ndarray) -> None:
        """入力データを検証

        Args:
            phi: Level Set関数の値

        Raises:
            TypeError: 入力が不適切な型の場合
            ValueError: 入力の次元が不適切な場合
        """
        if not isinstance(phi, np.ndarray):
            raise TypeError("入力はnumpy配列である必要があります")
        if phi.ndim not in [2, 3]:
            raise ValueError("2次元または3次元の配列である必要があります")
        if not np.isfinite(phi).all():
            raise ValueError("無限大または非数が含まれています")
