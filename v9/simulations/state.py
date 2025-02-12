"""シミュレーションの状態を管理するモジュール

このモジュールは、二相流シミュレーションの状態（速度場、レベルセット場、圧力場など）を
管理するためのデータクラスを提供します。
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.navier_stokes.terms import AccelerationTerm


@dataclass
class SimulationState:
    """シミュレーションの状態を保持するクラス

    物理量の場や時刻情報を保持し、状態の保存・読み込みも担当します。
    """

    time: float
    velocity: VectorField
    levelset: LevelSetField
    pressure: ScalarField
    diagnostics: Dict[str, Any] = None

    def __post_init__(self):
        """初期化後の処理"""
        if self.diagnostics is None:
            self.diagnostics = {}

    def validate(self) -> None:
        """状態の妥当性を検証"""
        if self.time < 0:
            raise ValueError("時刻は非負である必要があります")

        shapes = {
            "velocity": self.velocity.shape,
            "levelset": self.levelset.shape,
            "pressure": self.pressure.shape,
        }
        if len(set(shapes.values())) > 1:
            raise ValueError(f"場の形状が一致しません: {shapes}")

    def get_density(self) -> ScalarField:
        """密度場を計算"""
        density = ScalarField(self.levelset.shape, self.levelset.dx)
        density.data = self.levelset.get_heaviside().data
        return density

    def get_viscosity(self) -> ScalarField:
        """粘性場を計算"""
        viscosity = ScalarField(self.levelset.shape, self.levelset.dx)
        viscosity.data = self.levelset.get_heaviside().data
        return viscosity

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "time": self.time,
            "velocity_max": float(
                np.max([np.abs(v.data).max() for v in self.velocity.components])
            ),
            "pressure_max": float(np.abs(self.pressure.data).max()),
            "levelset_min": float(self.levelset.data.min()),
            "levelset_max": float(self.levelset.data.max()),
            "interface_geometry": self.levelset.get_geometry_info(),
            **self.diagnostics,
        }

    def save_state(self, filepath: str) -> None:
        """状態をファイルに保存

        Args:
            filepath: 保存先のファイルパス
        """
        np.savez_compressed(
            filepath,
            time=self.time,
            velocity_data=[v.data for v in self.velocity.components],
            levelset_data=self.levelset.data,
            pressure_data=self.pressure.data,
            diagnostics=self.diagnostics,
        )

    @classmethod
    def load_state(cls, filepath: str) -> "SimulationState":
        """ファイルから状態を読み込み

        Args:
            filepath: 読み込むファイルのパス

        Returns:
            読み込まれたシミュレーション状態
        """
        data = np.load(filepath)

        # 速度場の再構築
        velocity_shape = data["velocity_data"][0].shape
        velocity = VectorField(velocity_shape)
        for i, v_data in enumerate(data["velocity_data"]):
            velocity.components[i].data = v_data

        # レベルセット場の再構築
        levelset = LevelSetField(shape=data["levelset_data"].shape)
        levelset.data = data["levelset_data"]

        # 圧力場の再構築
        pressure = ScalarField(data["pressure_data"].shape)
        pressure.data = data["pressure_data"]

        return cls(
            time=float(data["time"]),
            velocity=velocity,
            levelset=levelset,
            pressure=pressure,
            diagnostics=data["diagnostics"].item(),
        )

    def compute_derivative(self) -> "SimulationState":
        """状態の時間微分を計算

        Returns:
            時間微分を表す新しい状態
        """
        # Level Set方程式の時間発展（界面の移流）
        levelset_derivative = self.levelset.compute_derivative(self.velocity)

        # 密度と粘性を計算
        density = self.get_density()
        viscosity = self.get_viscosity()

        # 加速度項を計算
        acceleration_term = AccelerationTerm()
        acceleration = acceleration_term.compute(
            velocity=self.velocity,
            density=density,
            viscosity=viscosity,
            pressure=self.pressure,
        )

        # 新しい状態を初期化
        derivative_state = SimulationState(
            time=0.0,
            velocity=VectorField(self.velocity.shape, self.velocity.dx),
            levelset=LevelSetField(shape=self.levelset.shape, dx=self.levelset.dx),
            pressure=ScalarField(self.pressure.shape, self.pressure.dx),
        )

        # 時間微分を設定
        derivative_state.levelset.data = levelset_derivative
        derivative_state.velocity.components = [
            ScalarField(v.shape, v.dx, initial_value=a)
            for v, a in zip(self.velocity.components, acceleration)
        ]

        return derivative_state
