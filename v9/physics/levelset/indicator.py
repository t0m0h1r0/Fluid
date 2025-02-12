"""Level Set法でのインジケーター関数を提供するモジュール

このモジュールは、Heaviside関数とDelta関数の実装を提供します。
これらの関数は、Level Set関数から物性値の分布を計算する際に使用されます。
"""

import numpy as np


class BaseIndicator:
    """インジケーター関数の基底クラス"""

    def __init__(self, epsilon: float = 1.0e-2):
        """
        Args:
            epsilon: 界面の厚さパラメータ
        """
        if epsilon <= 0:
            raise ValueError("epsilonは正の値である必要があります")
        self.epsilon = epsilon

    def compute(self, phi: np.ndarray) -> np.ndarray:
        """インジケーター関数を計算

        Args:
            phi: Level Set関数の値

        Returns:
            計算されたインジケーター関数の値
        """
        raise NotImplementedError


class HeavisideFunction(BaseIndicator):
    """正則化されたHeaviside関数"""

    def compute(self, phi: np.ndarray) -> np.ndarray:
        """Heaviside関数を計算: H(ϕ)

        Args:
            phi: Level Set関数の値

        Returns:
            計算されたHeaviside関数の値
        """
        # tanh関数を用いた正則化
        return 0.5 * (1.0 + np.tanh(phi / self.epsilon))


class DeltaFunction(BaseIndicator):
    """正則化されたDelta関数"""

    def compute(self, phi: np.ndarray) -> np.ndarray:
        """Delta関数を計算: δ(ϕ)

        Args:
            phi: Level Set関数の値

        Returns:
            計算されたDelta関数の値
        """
        # Heaviside関数の導関数として計算
        return 0.5 / self.epsilon * (1.0 - np.tanh(phi / self.epsilon) ** 2)
