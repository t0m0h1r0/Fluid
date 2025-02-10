"""Level Set場を定義するモジュール

このモジュールは、Level Set法で使用される場のクラスを提供します。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np

from .utils import (
    compute_volume,
    compute_area,
    heaviside,
    delta,
    compute_curvature,
    compute_interface_gradient,
)


@dataclass
class LevelSetParameters:
    """Level Set法のパラメータ"""

    epsilon: float = 1.0e-2  # 界面の厚さ
    min_value: float = 1.0e-10  # 最小値
    reinit_interval: int = 5  # 再初期化の間隔
    reinit_steps: int = 2  # 再初期化のステップ数
    reinit_dt: float = 0.1  # 再初期化の時間刻み


class LevelSetField:
    """Level Set場クラス

    Level Set関数を表現し、界面追跡のための基本的な操作を提供します。
    """

    def __init__(
        self,
        shape: Union[Tuple[int, ...], np.ndarray],
        dx: float = 1.0,
        params: Optional[LevelSetParameters] = None,
    ):
        """Level Set場を初期化

        Args:
            shape: グリッドの形状またはデータ配列
            dx: グリッド間隔
            params: Level Setパラメータ
        """
        # shapeがタプルの場合は新しい配列を作成
        if isinstance(shape, tuple):
            self._data = np.zeros(shape)
        # ndarrayの場合はコピーを作成
        else:
            self._data = shape.copy()

        self._dx = dx
        self.params = params or LevelSetParameters()

        # ステップ管理
        self._steps_since_reinit = 0
        self._initial_volume = self.compute_volume()

    @property
    def data(self) -> np.ndarray:
        """Level Set関数のデータを取得"""
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Level Set関数のデータを設定

        Args:
            value: 設定するデータ配列

        Raises:
            ValueError: データの形状が不一致の場合
        """
        if isinstance(value, np.ndarray):
            if value.shape != self._data.shape:
                raise ValueError(
                    f"形状が一致しません: {value.shape} != {self._data.shape}"
                )
            self._data = value.copy()
        else:
            self._data = np.asarray(value)

    @property
    def shape(self) -> Tuple[int, ...]:
        """グリッドの形状を取得"""
        return self._data.shape

    @property
    def dx(self) -> float:
        """グリッド間隔を取得"""
        return self._dx

    @property
    def ndim(self) -> int:
        """次元数を取得"""
        return self._data.ndim

    def heaviside(self) -> np.ndarray:
        """Heaviside関数の値を計算

        Returns:
            Heaviside関数の値
        """
        return heaviside(self._data, self.params.epsilon)

    def delta(self) -> np.ndarray:
        """Delta関数の値を計算

        Returns:
            Delta関数の値
        """
        return delta(self._data, self.params.epsilon)

    def curvature(self) -> np.ndarray:
        """界面の曲率を計算

        Returns:
            界面の曲率
        """
        return compute_curvature(self._data, self._dx)

    def compute_volume(self) -> float:
        """体積を計算

        Returns:
            計算された体積
        """
        return compute_volume(self._data, self._dx)

    def compute_area(self) -> float:
        """界面の面積を計算

        Returns:
            計算された面積
        """
        return compute_area(self._data, self._dx)

    def compute_interface_gradient(self) -> np.ndarray:
        """界面の法線ベクトルを計算

        Returns:
            界面の法線ベクトル
        """
        return compute_interface_gradient(self._data, self._dx)

    def gradient(self, axis: int) -> np.ndarray:
        """指定軸方向の勾配を計算

        Args:
            axis: 勾配を計算する軸のインデックス

        Returns:
            計算された勾配
        """
        return np.gradient(self._data, self._dx, axis=axis)

    def need_reinit(self) -> bool:
        """再初期化が必要かどうかを判定

        Returns:
            再初期化が必要かどうか
        """
        return (
            self._steps_since_reinit >= self.params.reinit_interval
            or not self._is_signed_distance_function()
        )

    def _is_signed_distance_function(self, tolerance: float = 1e-2) -> bool:
        """符号付き距離関数としての性質を検証

        Args:
            tolerance: 許容誤差

        Returns:
            符号付き距離関数の条件を満たすかどうか
        """
        # 勾配の大きさが1に近いかチェック
        grad = np.gradient(self._data, self._dx)
        grad_norm = np.sqrt(sum(g**2 for g in grad))

        # 勾配の大きさが1にどれだけ近いか
        is_unit_gradient = np.abs(grad_norm - 1.0)

        # 界面の幅をチェック
        interface_width = np.sum(self.delta() > 0) * self._dx**self.ndim

        # 両条件を確認
        return np.mean(is_unit_gradient) < tolerance and interface_width < tolerance

    def advance_step(self):
        """時間ステップを進める"""
        self._steps_since_reinit += 1

    def get_interface_points(self) -> np.ndarray:
        """界面上の点を取得

        Returns:
            界面上の点の座標
        """
        # Delta関数を使用して界面上の点を抽出
        interface_mask = self.delta() > 0
        return np.argwhere(interface_mask)

    def copy(self) -> "LevelSetField":
        """フィールドの深いコピーを作成

        Returns:
            コピーされたLevel Set場
        """
        return LevelSetField(data=self._data.copy(), dx=self._dx, params=self.params)
