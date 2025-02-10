"""Level Set場を定義するモジュール

このモジュールは、Level Set法で使用される場のクラスを提供します。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Union
from core.field import ConservedField
from .config import LevelSetConfig
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

    epsilon: float = 1.0e-2
    min_value: float = 1.0e-10
    reinit_interval: int = 5
    reinit_steps: int = 2
    reinit_dt: float = 0.1


class LevelSetField(ConservedField):
    """Level Set場クラス

    Level Set関数を表現し、界面追跡のための基本的な操作を提供します。
    """

    def __init__(
        self,
        shape: tuple,
        dx: float = 1.0,
        config: Optional[LevelSetConfig] = None,
        params: Optional[LevelSetParameters] = None,
    ):
        """Level Set場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔
            config: Level Set設定
            params: Level Setパラメータ
        """
        # パラメータを先に設定
        self.config = config or LevelSetConfig()
        self.params = params or LevelSetParameters()

        # ステップ管理
        self._steps_since_reinit = 0
        self._initial_volume = None

        # 親クラスの初期化
        super().__init__(shape, dx)

        # 初期体積を記録
        self._initial_volume = self.compute_volume()

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
        return compute_curvature(self._data, self.dx)

    def compute_volume(self) -> float:
        """体積を計算

        Returns:
            計算された体積
        """
        return compute_volume(self._data, self.dx, self.params.epsilon)

    def compute_area(self) -> float:
        """界面の面積を計算

        Returns:
            計算された面積
        """
        return compute_area(self._data, self.dx, self.params.epsilon)

    def compute_interface_gradient(self) -> np.ndarray:
        """界面の法線ベクトルを計算

        Returns:
            界面の法線ベクトル
        """
        return compute_interface_gradient(self._data, self.dx)

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
        grad = np.gradient(self._data, self.dx)
        grad_norm = np.sqrt(sum(g**2 for g in grad))

        # 勾配の大きさが1にどれだけ近いか
        is_unit_gradient = np.abs(grad_norm - 1.0)

        # 界面の幅をチェック
        interface_width = np.sum(self.delta() > 0) * self.dx**self.ndim

        # 両条件を確認
        return np.mean(is_unit_gradient) < tolerance and interface_width < tolerance

    def advance_step(self):
        """時間ステップを進める"""
        self._steps_since_reinit += 1

    def get_values_at_interface(self, field: np.ndarray) -> np.ndarray:
        """界面上での物理量の値を取得

        Args:
            field: 値を取得する物理量

        Returns:
            界面上の値
        """
        return self.delta() * field

    def get_diagnostics(self) -> Dict[str, Union[float, int]]:
        """診断情報を取得

        Returns:
            診断情報の辞書
        """
        config_diagnostics = self.config.get_config_for_component("diagnostics")

        diagnostics = {
            "volume": None,
            "area": None,
            "volume_ratio": None,
            "steps_since_reinit": self._steps_since_reinit,
            "min_value": float(np.min(self._data)),
            "max_value": float(np.max(self._data)),
        }

        if config_diagnostics.get("compute_volume", True):
            diagnostics["volume"] = self.compute_volume()
            diagnostics["volume_ratio"] = diagnostics["volume"] / (
                self._initial_volume or 1.0
            )

        if config_diagnostics.get("compute_area", True):
            diagnostics["area"] = self.compute_area()

        return diagnostics

    def __str__(self) -> str:
        """文字列表現を取得

        Returns:
            Level Set場の文字列表現
        """
        diag = self.get_diagnostics()
        return (
            f"LevelSetField:\n"
            f"  Volume: {diag.get('volume', 'N/A')}\n"
            f"  Area: {diag.get('area', 'N/A')}\n"
            f"  Volume Ratio: {diag.get('volume_ratio', 'N/A')}\n"
            f"  Value Range: [{diag['min_value']}, {diag['max_value']}]\n"
            f"  Steps Since Reinit: {diag['steps_since_reinit']}"
        )
