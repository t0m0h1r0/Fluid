"""Level Set場を提供するモジュール"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
from core.field import ConservedField
from .utils import (
    heaviside,
    delta,
    compute_curvature,
    reinitialize,
    compute_volume,
    compute_area,
)


@dataclass
class LevelSetParameters:
    """Level Set法のパラメータ"""

    epsilon: float = 1.0e-2
    min_value: float = 1.0e-10
    reinit_interval: int = 1
    reinit_steps: int = 5
    reinit_dt: float = 0.1


class LevelSetField(ConservedField):
    """Level Set場クラス"""

    def __init__(
        self, shape: tuple, dx: float = 1.0, params: Optional[LevelSetParameters] = None
    ):
        """Level Set場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔
            params: Level Set法のパラメータ
        """
        # パラメータを先に設定
        self.params = params or LevelSetParameters()
        self._steps_since_reinit = 0

        # 親クラスの初期化
        super().__init__(shape, dx)

        # 初期体積を記録
        self._initial_volume = self.compute_volume()

    def heaviside(self) -> np.ndarray:
        """Heaviside関数の値を計算"""
        return heaviside(self._data, self.params.epsilon)

    def delta(self) -> np.ndarray:
        """Delta関数の値を計算"""
        return delta(self._data, self.params.epsilon)

    def curvature(self) -> np.ndarray:
        """界面の曲率を計算"""
        return compute_curvature(self._data, self.dx)

    def compute_volume(self) -> float:
        """体積を計算"""
        return compute_volume(self._data, self.dx, self.params.epsilon)

    def compute_area(self) -> float:
        """界面の面積を計算"""
        return compute_area(self._data, self.dx, self.params.epsilon)

    def need_reinit(self) -> bool:
        """再初期化が必要かどうかを判定"""
        return self._steps_since_reinit >= self.params.reinit_interval

    def reinitialize(self):
        """Level Set関数を再初期化"""
        if not self.need_reinit():
            return

        # 前の体積を記録
        old_volume = self.compute_volume()

        # 再初期化
        self._data = reinitialize(
            self._data,
            self.dx,
            dt=self.params.reinit_dt,
            n_steps=self.params.reinit_steps,
        )

        # 体積を保存するように補正
        new_volume = self.compute_volume()
        if abs(new_volume) > self.params.min_value:
            self._data *= (old_volume / new_volume) ** (1.0 / self.ndim)

        self._steps_since_reinit = 0

    def integrate(self) -> float:
        """保存量（体積）を計算"""
        return self.compute_volume()

    def advance_step(self):
        """時間ステップを進める"""
        self._steps_since_reinit += 1

    def get_values_at_interface(self, field: np.ndarray) -> np.ndarray:
        """界面上での物理量の値を取得"""
        return self.delta() * field

    def get_diagnostics(self) -> Dict[str, float]:
        """診断情報を取得"""
        return {
            "volume": self.compute_volume(),
            "area": self.compute_area(),
            "volume_ratio": self.compute_volume() / self._initial_volume,
            "min_value": np.min(self._data),
            "max_value": np.max(self._data),
            "steps_since_reinit": self._steps_since_reinit,
        }

    def __str__(self) -> str:
        """文字列表現"""
        diag = self.get_diagnostics()
        return (
            f"LevelSetField:\n"
            f"  Volume: {diag['volume']:.6g}\n"
            f"  Area: {diag['area']:.6g}\n"
            f"  Volume Ratio: {diag['volume_ratio']:.6g}\n"
            f"  Value Range: [{diag['min_value']:.6g}, {diag['max_value']:.6g}]"
        )
