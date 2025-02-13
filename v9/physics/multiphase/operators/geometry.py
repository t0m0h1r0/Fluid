"""界面の幾何学的演算を提供するモジュール

法線ベクトルや曲率などの幾何学的量の計算を実装します。
"""

import numpy as np
from core.field import VectorField, ScalarField


class GeometryOperator:
    """界面の幾何学的演算を実行するクラス"""

    def __init__(self, dx: np.ndarray, epsilon: float = 1.0e-6):
        """幾何演算子を初期化

        Args:
            dx: グリッド間隔（ベクトル）
            epsilon: 数値計算の安定化パラメータ
        """
        self.dx = dx
        self.epsilon = epsilon

    def compute_normal(self, phi: ScalarField) -> VectorField:
        """法線ベクトルを計算: n = ∇φ/|∇φ|

        Args:
            phi: 距離関数

        Returns:
            法線ベクトル場
        """
        # 勾配の計算
        grad_components = [phi.gradient(i) for i in range(phi.ndim)]

        # 勾配ノルムの計算
        grad_norm = np.sqrt(sum(g * g for g in grad_components))
        grad_norm = np.maximum(grad_norm, self.epsilon)

        # 法線ベクトル場の構築
        normal = VectorField(phi.shape, self.dx)
        for i in range(phi.ndim):
            normal.components[i] = ScalarField(
                phi.shape, self.dx, initial_value=grad_components[i] / grad_norm
            )

        return normal

    def compute_curvature(
        self, phi: ScalarField, method: str = "standard"
    ) -> ScalarField:
        """曲率を計算: κ = ∇⋅n = ∇⋅(∇φ/|∇φ|)

        Args:
            phi: 距離関数
            method: 計算手法 ('standard' または 'high_order')

        Returns:
            曲率場
        """
        if method == "standard":
            return self._compute_standard_curvature(phi)
        elif method == "high_order":
            return self._compute_high_order_curvature(phi)
        else:
            raise ValueError(f"未知の計算手法: {method}")

    def _compute_standard_curvature(self, phi: ScalarField) -> ScalarField:
        """2次精度の標準的な曲率計算"""
        # 勾配と勾配ノルムの計算
        grad_components = [phi.gradient(i) for i in range(phi.ndim)]
        grad_norm = np.sqrt(sum(g * g for g in grad_components))
        grad_norm = np.maximum(grad_norm, self.epsilon)

        # 法線ベクトルの発散を計算
        result = ScalarField(phi.shape, self.dx)
        for i in range(phi.ndim):
            normalized_grad = grad_components[i] / grad_norm
            # 各方向の勾配を個別に計算
            result.data += np.gradient(normalized_grad, self.dx[i], axis=i)

        return result

    def _compute_high_order_curvature(self, phi: ScalarField) -> ScalarField:
        """4次精度の高精度曲率計算"""
        result = ScalarField(phi.shape, self.dx)
        ndim = phi.ndim

        for i in range(ndim):
            for j in range(ndim):
                if i == j:
                    # 対角項の4次精度差分
                    d2 = self._compute_4th_order_derivative(phi, i, i)
                else:
                    # 交差微分項の4次精度差分
                    d2 = self._compute_4th_order_cross_derivative(phi, i, j)

                result.data += d2

        return result

    def _compute_4th_order_derivative(
        self, phi: ScalarField, i: int, j: int
    ) -> np.ndarray:
        """4次精度の2階微分を計算"""
        data = phi.data
        dx = self.dx[i]  # 対応する方向のグリッド間隔を使用

        # 4次精度の係数
        c = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]

        # 2階微分の計算
        result = np.zeros_like(data)
        for k in range(5):
            shifted = np.roll(data, k - 2, axis=i)
            result += c[k] * shifted

        return result / (dx * dx)

    def _compute_4th_order_cross_derivative(
        self, phi: ScalarField, i: int, j: int
    ) -> np.ndarray:
        """4次精度の交差微分を計算"""
        data = phi.data
        dx_i = self.dx[i]  # i方向のグリッド間隔
        dx_j = self.dx[j]  # j方向のグリッド間隔

        # まず i 方向の1階微分を計算
        di = np.zeros_like(data)
        for k in range(-2, 3):
            shifted = np.roll(data, k, axis=i)
            di += (-1 / 12 if abs(k) == 2 else 2 / 3 if abs(k) == 1 else 0) * shifted
        di /= dx_i

        # 次に j 方向の1階微分を計算
        result = np.zeros_like(data)
        for k in range(-2, 3):
            shifted = np.roll(di, k, axis=j)
            result += (
                -1 / 12 if abs(k) == 2 else 2 / 3 if abs(k) == 1 else 0
            ) * shifted
        result /= dx_j

        return result
