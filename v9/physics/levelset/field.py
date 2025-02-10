from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt

from .utils import compute_volume, compute_area, heaviside, delta, compute_curvature


@dataclass
class LevelSetParameters:
    """Level Set法のパラメータを管理するデータクラス"""

    epsilon: float = 1.0e-2  # 界面の厚さ
    reinit_interval: int = 5  # 再初期化の間隔
    reinit_steps: int = 2  # 再初期化のステップ数
    min_value: float = 1.0e-10  # 最小値の閾値


class LevelSetField:
    """Level Set関数を表現し、界面追跡のための操作を提供するクラス"""

    def __init__(
        self,
        data: npt.NDArray,
        dx: float = 1.0,
        params: Optional[LevelSetParameters] = None,
    ):
        """
        Level Set関数を初期化

        Args:
            data: Level Set関数の数値データ
            dx: グリッド間隔
            params: Level Set法のパラメータ
        """
        self._data = np.asarray(data)
        self._dx = dx
        self.params = params or LevelSetParameters()
        self._steps_since_reinit = 0

    @property
    def data(self) -> npt.NDArray:
        """Level Set関数のデータを取得"""
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        """グリッドの形状を取得"""
        return self._data.shape

    @property
    def dx(self) -> float:
        """グリッド間隔を取得"""
        return self._dx

    def volume(self) -> float:
        """体積を計算"""
        return compute_volume(self._data, self._dx)

    def area(self) -> float:
        """界面の面積を計算"""
        return compute_area(self._data, self._dx)

    def curvature(self) -> npt.NDArray:
        """界面の曲率を計算"""
        return compute_curvature(self._data, self._dx)

    def heaviside(self) -> npt.NDArray:
        """Heaviside関数を計算"""
        return heaviside(self._data, self.params.epsilon)

    def delta(self) -> npt.NDArray:
        """Delta関数を計算"""
        return delta(self._data, self.params.epsilon)

    def needs_reinit(self) -> bool:
        """再初期化が必要かどうかを判定"""
        return (
            self._steps_since_reinit >= self.params.reinit_interval
            or not self._is_signed_distance()
        )

    def _is_signed_distance(self, tolerance: float = 1e-2) -> bool:
        """符号付き距離関数としての性質を検証"""
        grad = np.gradient(self._data, self._dx)
        grad_norm = np.sqrt(sum(g**2 for g in grad))
        return np.abs(grad_norm - 1.0).mean() < tolerance

    def copy(self) -> "LevelSetField":
        """深いコピーを作成"""
        return LevelSetField(data=self._data.copy(), dx=self._dx, params=self.params)
