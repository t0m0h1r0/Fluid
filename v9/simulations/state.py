"""シミュレーション状態を管理するモジュール

このモジュールは、流体シミュレーションの状態（速度場、圧力場、Level Set場など）を
管理するためのクラスを提供します。
"""

from typing import Dict, Any
from pathlib import Path
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField


class SimulationState:
    """シミュレーション状態クラス"""

    def __init__(self, shape: tuple, dx: float = 1.0):
        """シミュレーション状態を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔
        """
        # 速度場の初期化
        self._velocity = VectorField(shape, dx)

        # 圧力場の初期化
        self._pressure = ScalarField(shape, dx)

        # Level Set場の初期化
        self._levelset = LevelSetField(shape, dx)

        # 物性値マネージャーは後で設定される
        self._properties = None

        self._time = 0.0

    @property
    def velocity(self) -> VectorField:
        """速度場を取得"""
        return self._velocity

    @velocity.setter
    def velocity(self, value: VectorField):
        """速度場を設定"""
        if value.shape != self._velocity.shape:
            raise ValueError("速度場の形状が一致しません")
        self._velocity = value

    @property
    def pressure(self) -> ScalarField:
        """圧力場を取得"""
        return self._pressure

    @pressure.setter
    def pressure(self, value: ScalarField):
        """圧力場を設定"""
        if value.shape != self._pressure.shape:
            raise ValueError("圧力場の形状が一致しません")
        self._pressure = value

    @property
    def levelset(self) -> LevelSetField:
        """Level Set場を取得"""
        return self._levelset

    @levelset.setter
    def levelset(self, value: LevelSetField):
        """Level Set場を設定"""
        if value.shape != self._levelset.shape:
            raise ValueError("Level Set場の形状が一致しません")
        self._levelset = value

    @property
    def properties(self):
        """物性値マネージャーを取得"""
        return self._properties

    @properties.setter
    def properties(self, value):
        """物性値マネージャーを設定"""
        self._properties = value

    @property
    def time(self) -> float:
        """現在の時刻を取得"""
        return self._time

    @time.setter
    def time(self, value: float):
        """時刻を設定"""
        if value < 0:
            raise ValueError("時刻は非負である必要があります")
        self._time = value

    def save_state(self) -> Dict[str, Any]:
        """状態を保存

        Returns:
            現在の状態を表す辞書
        """
        return {
            "velocity": self.velocity.save_state(),
            "pressure": self.pressure.save_state(),
            "levelset": self.levelset.save_state(),
            "time": self._time,
        }

    def load_state(self, state: Dict[str, Any]):
        """状態を読み込み

        Args:
            state: 読み込む状態の辞書
        """
        self.velocity.load_state(state["velocity"])
        self.pressure.load_state(state["pressure"])
        self.levelset.load_state(state["levelset"])
        self._time = state["time"]

    def save_to_file(self, filename: str):
        """状態をファイルに保存

        Args:
            filename: 保存するファイル名
        """
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            filename,
            velocity_x=self.velocity.components[0].data,
            velocity_y=self.velocity.components[1].data,
            velocity_z=self.velocity.components[2].data,
            pressure=self.pressure.data,
            levelset=self.levelset.data,
            time=self._time,
        )

    @classmethod
    def load_from_file(cls, filename: str, shape: tuple = None, dx: float = 1.0):
        """ファイルから状態を読み込み

        Args:
            filename: 読み込むファイル名
            shape: グリッドの形状（指定がない場合はファイルから読み取り）
            dx: グリッド間隔

        Returns:
            読み込まれた状態
        """
        data = np.load(filename)

        if shape is None:
            shape = data["velocity_x"].shape

        state = cls(shape, dx)
        state.velocity.components[0].data = data["velocity_x"]
        state.velocity.components[1].data = data["velocity_y"]
        state.velocity.components[2].data = data["velocity_z"]
        state.pressure.data = data["pressure"]
        state.levelset.data = data["levelset"]
        state._time = float(data["time"])

        return state

    def get_extrema(self) -> Dict[str, Dict[str, float]]:
        """各フィールドの最大値・最小値を取得

        Returns:
            各フィールドの最大値・最小値を含む辞書
        """
        return {
            "velocity": {
                "min": min(np.min(c.data) for c in self.velocity.components),
                "max": max(np.max(c.data) for c in self.velocity.components),
            },
            "pressure": {
                "min": np.min(self.pressure.data),
                "max": np.max(self.pressure.data),
            },
            "levelset": {
                "min": np.min(self.levelset.data),
                "max": np.max(self.levelset.data),
            },
        }

    def __str__(self) -> str:
        """文字列表現"""
        extrema = self.get_extrema()
        return (
            f"SimulationState (t = {self._time:.3f}):\n"
            f"  Velocity: [{extrema['velocity']['min']:.3f}, {extrema['velocity']['max']:.3f}]\n"
            f"  Pressure: [{extrema['pressure']['min']:.3f}, {extrema['pressure']['max']:.3f}]\n"
            f"  Level Set: [{extrema['levelset']['min']:.3f}, {extrema['levelset']['max']:.3f}]"
        )
