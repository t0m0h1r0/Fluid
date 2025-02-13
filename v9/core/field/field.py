"""場の基底クラスを提供するモジュール

このモジュールは流体シミュレーションで使用される場の基底クラスを定義します。
すべての具体的な場の実装（スカラー場、ベクトル場など）は、この基底クラスを継承します。
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class Field(ABC):
    """場の基底クラス

    この抽象基底クラスは、流体シミュレーションで使用される全ての場に共通の
    インターフェースと基本機能を提供します。

    Attributes:
        shape (Tuple[int, ...]): グリッドの形状
        dx (float): グリッド間隔
        time (float): 現在の時刻
    """

    def __init__(self, shape: Tuple[int, ...], dx: float = 1.0):
        """場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔
        """
        self._data = np.zeros(shape)
        self._dx = dx
        self._time = 0.0

    @property
    def data(self) -> np.ndarray:
        """場のデータを取得"""
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """場のデータを設定

        Args:
            value: 設定するデータ配列

        Raises:
            ValueError: データの形状が不一致の場合
        """
        if value.shape != self._data.shape:
            raise ValueError(f"形状が一致しません: {value.shape} != {self._data.shape}")
        self._data = value.copy()  # 防御的コピー

    @property
    def shape(self) -> Tuple[int, ...]:
        """場の形状を取得"""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """次元数を取得"""
        return self._data.ndim

    @property
    def dx(self) -> float:
        """グリッド間隔を取得"""
        return self._dx

    @property
    def time(self) -> float:
        """現在の時刻を取得"""
        return self._time

    @time.setter
    def time(self, value: float):
        """時刻を設定

        Args:
            value: 設定する時刻

        Raises:
            ValueError: 負の時刻が指定された場合
        """
        if value < 0:
            raise ValueError("時刻は非負である必要があります")
        self._time = value

    def gradient(self, axis: int) -> np.ndarray:
        """指定軸方向の勾配を計算

        中心差分による2次精度の勾配計算を行います。

        Args:
            axis: 勾配を計算する軸のインデックス

        Returns:
            計算された勾配
        """
        return np.gradient(self._data, self._dx, axis=axis)

    def divergence(self) -> np.ndarray:
        """発散を計算

        中心差分による2次精度の発散計算を行います。

        Returns:
            計算された発散
        """
        div = np.zeros_like(self._data)
        for i in range(self.ndim):
            div += np.gradient(self._data, self._dx, axis=i)
        return div

    def laplacian(self) -> np.ndarray:
        """ラプラシアンを計算

        中心差分による2次精度のラプラシアン計算を行います。

        Returns:
            計算されたラプラシアン
        """
        return np.gradient(self.divergence(), self._dx)

    def copy(self) -> "Field":
        """場の深いコピーを作成

        Returns:
            コピーされた場
        """
        new_field = self.__class__(self.shape, self.dx)
        new_field.data = self.data.copy()
        new_field.time = self.time
        return new_field

    def save_state(self) -> Dict[str, Any]:
        """現在の状態を保存

        Returns:
            現在の状態を表す辞書
        """
        return {"data": self._data.copy(), "time": self._time, "dx": self._dx}

    def load_state(self, state: Dict[str, Any]):
        """状態を読み込み

        Args:
            state: 読み込む状態の辞書
        """
        self._data = state["data"].copy()
        self._time = state["time"]
        self._dx = state["dx"]

    @abstractmethod
    def integrate(self) -> float:
        """場の積分値を計算

        Returns:
            計算された積分値
        """
        pass