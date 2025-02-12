"""Level Set関数の再初期化を提供するモジュール"""

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import BaseLevelSetOperation


class LevelSetReinitializer(BaseLevelSetOperation):
    """Level Set関数の再初期化クラス"""

    def __init__(self, dx: float, epsilon: float = 1.0e-2):
        """
        Args:
            dx: グリッド間隔
            epsilon: 界面の厚さパラメータ
        """
        super().__init__(dx)
        self.epsilon = epsilon

    def reinitialize(
        self, phi: np.ndarray, n_steps: int = 5, dt: float = 0.1
    ) -> np.ndarray:
        """Level Set関数を再初期化

        Args:
            phi: Level Set関数の値
            n_steps: 反復回数
            dt: 時間刻み幅

        Returns:
            再初期化されたLevel Set関数の値
        """
        # 入力の検証
        self.validate_input(phi)

        # 結果の初期化
        result = phi.copy()
        sign = np.sign(result)

        # 高速行進法による再初期化
        for _ in range(n_steps):
            # 界面近傍の点を特定
            interface_points = np.abs(result) < self.epsilon

            # 勾配を計算
            grad = np.array(np.gradient(result, self.dx))
            grad_norm = np.sqrt(np.sum(grad**2, axis=0))

            # 時間発展
            correction = sign * (grad_norm - 1.0)
            result -= dt * correction

            # 数値的安定化のためにガウシアンフィルタを適用
            result = gaussian_filter(result, sigma=0.5 * self.dx)

        return result

    def validate_input(self, phi: np.ndarray) -> None:
        """入力データを検証

        Args:
            phi: Level Set関数の値
        """
        if not isinstance(phi, np.ndarray):
            raise TypeError("入力はnumpy配列である必要があります")
        if phi.ndim not in [2, 3]:
            raise ValueError("2次元または3次元の配列である必要があります")
