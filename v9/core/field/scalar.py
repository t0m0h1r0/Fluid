"""スカラー場クラスを提供するモジュール

このモジュールは、スカラー量（圧力、温度など）を表現するための場のクラスを定義します。
"""

from typing import Tuple, Optional
import numpy as np
from .field import Field


class ScalarField(Field):
    """スカラー場クラス

    温度、圧力などのスカラー量を表現するためのクラスです。
    基本的な微分演算や補間機能を提供します。
    """

    def __init__(
        self, shape: Tuple[int, ...], dx: float = 1.0, initial_value: float = 0.0
    ):
        """スカラー場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔
            initial_value: 初期値
        """
        super().__init__(shape, dx)
        if initial_value != 0.0:
            self._data.fill(initial_value)

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        """任意の点での値を線形補間

        Args:
            points: 補間点の座標 (N, ndim)

        Returns:
            補間された値 (N,)
        """
        # 各次元のインデックスと重みを計算
        indices = []
        weights = []

        for dim in range(self.ndim):
            # 座標をインデックスに変換
            idx = points[:, dim] / self._dx
            idx0 = np.floor(idx).astype(int)
            idx1 = idx0 + 1
            w1 = idx - idx0
            w0 = 1.0 - w1

            # 境界条件の適用
            idx0 = np.clip(idx0, 0, self.shape[dim] - 1)
            idx1 = np.clip(idx1, 0, self.shape[dim] - 1)

            indices.append((idx0, idx1))
            weights.append((w0, w1))

        # 全ての隣接点での重み付き和を計算
        result = np.zeros(len(points))
        for i in range(2**self.ndim):
            # i のビット表現から各次元でのインデックスを決定
            idx = []
            w = 1.0
            for d in range(self.ndim):
                bit = (i >> d) & 1
                idx.append(indices[d][bit])
                w *= weights[d][bit]

            # インデックスでの値を重み付きで加算
            result += w * self._data[tuple(idx)]

        return result

    def integrate(self) -> float:
        """場の積分値を計算"""
        return np.sum(self._data) * self._dx**self.ndim

    def mean(self) -> float:
        """場の平均値を計算"""
        return np.mean(self._data)

    def min(self) -> float:
        """場の最小値を取得"""
        return np.min(self._data)

    def max(self) -> float:
        """場の最大値を取得"""
        return np.max(self._data)

    def normalize(self):
        """場を正規化

        場の値を[0, 1]の範囲に正規化します。
        """
        min_val = self.min()
        max_val = self.max()
        if max_val > min_val:
            self._data = (self._data - min_val) / (max_val - min_val)

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        """場の値を指定範囲に制限

        Args:
            min_val: 最小値（Noneの場合は制限なし）
            max_val: 最大値（Noneの場合は制限なし）
        """
        self._data = np.clip(self._data, min_val, max_val)

    def smooth(self, sigma: float = 1.0):
        """場をガウシアンフィルタで平滑化

        Args:
            sigma: ガウシアンフィルタの標準偏差
        """
        from scipy.ndimage import gaussian_filter

        self._data = gaussian_filter(self._data, sigma)

    def __add__(self, other):
        """加算演算子の実装"""
        result = self.__class__(self.shape, self.dx)
        if isinstance(other, (int, float)):
            result.data = self.data + other
        elif isinstance(other, ScalarField):
            if self.shape != other.shape:
                raise ValueError("場の形状が一致しません")
            result.data = self.data + other.data
        else:
            raise TypeError("無効な型との演算です")
        return result

    def __mul__(self, other):
        """乗算演算子の実装"""
        result = self.__class__(self.shape, self.dx)
        if isinstance(other, (int, float)):
            result.data = self.data * other
        elif isinstance(other, ScalarField):
            if self.shape != other.shape:
                raise ValueError("場の形状が一致しません")
            result.data = self.data * other.data
        else:
            raise TypeError("無効な型との演算です")
        return result

    def __rmul__(self, other):
        """右乗算演算子の実装"""
        return self.__mul__(other)

    def __truediv__(self, other):
        """除算演算子の実装"""
        result = self.__class__(self.shape, self.dx)
        if isinstance(other, (int, float)):
            result.data = self.data / other
        elif isinstance(other, ScalarField):
            if self.shape != other.shape:
                raise ValueError("場の形状が一致しません")
            result.data = self.data / other.data
        else:
            raise TypeError("無効な型との演算です")
        return result
