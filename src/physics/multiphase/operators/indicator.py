"""相識別関数を提供するモジュール

このモジュールは、界面を通じた相の分布を表現するための
Heaviside関数とDelta関数を実装します。
"""

import numpy as np
from core.field import ScalarField


class IndicatorOperator:
    """相識別関数を計算するクラス"""

    def __init__(self, epsilon: float = 1.0e-2):
        """相識別演算子を初期化

        Args:
            epsilon: 界面の厚さパラメータ
        """
        self.epsilon = epsilon

    def compute_heaviside(self, phi: ScalarField) -> ScalarField:
        """Heaviside関数を計算: H(φ)

        Args:
            phi: 距離関数

        Returns:
            正則化されたHeaviside関数
        """
        result = ScalarField(phi.shape, phi.dx)
        # tanh関数を用いた正則化
        result.data = 0.5 * (1.0 + np.tanh(phi.data / self.epsilon))
        return result

    def compute_delta(self, phi: ScalarField) -> ScalarField:
        """Delta関数を計算: δ(φ)

        Args:
            phi: 距離関数

        Returns:
            正則化されたDelta関数
        """
        result = ScalarField(phi.shape, phi.dx)
        # Heaviside関数の導関数として計算
        result.data = 0.5 / self.epsilon * (1.0 - np.tanh(phi.data / self.epsilon) ** 2)
        return result

    def get_phase_field(
        self, phi: ScalarField, value1: float, value2: float
    ) -> ScalarField:
        """物性値の分布を計算: f = f₁H(φ) + f₂(1-H(φ))

        Args:
            phi: 距離関数
            value1: 第1相の物性値
            value2: 第2相の物性値

        Returns:
            物性値の分布
        """
        result = ScalarField(phi.shape, phi.dx)
        h = self.compute_heaviside(phi)
        result.data = value1 * h.data + value2 * (1.0 - h.data)
        return result

    def get_interface_measure(self, phi: ScalarField) -> float:
        """界面の面積/長さを計算

        Args:
            phi: 距離関数

        Returns:
            界面の面積（3D）または長さ（2D）
        """
        delta = self.compute_delta(phi)
        # グリッド体積要素を計算（非等方グリッドに対応）
        dv = np.prod(phi.dx)
        return float(np.sum(delta.data) * dv)
