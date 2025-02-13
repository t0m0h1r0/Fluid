"""
シミュレーションの状態を管理するモジュール

このモジュールは、二相流シミュレーションの状態（速度場、界面関数、圧力場など）を
管理するためのデータクラスを提供します。
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from physics.multiphase import InterfaceOperations


@dataclass
class SimulationState:
    """シミュレーションの状態を保持するクラス

    物理量の場や時刻情報を保持し、状態の保存・読み込みも担当します。
    """

    time: float
    velocity: VectorField
    levelset: ScalarField  # ScalarFieldとして定義
    pressure: ScalarField
    diagnostics: Dict[str, Any] = None

    def __post_init__(self):
        """初期化後の処理"""
        if self.diagnostics is None:
            self.diagnostics = {}

        # 界面演算子の初期化
        self._interface_ops = InterfaceOperations(
            dx=self.velocity.dx,
            epsilon=1e-2,  # デフォルトのepsilon値
        )

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

    def get_phase_distribution(self) -> ScalarField:
        """密度場を計算"""
        # InterfaceOperationsのメソッドを使用
        return self._interface_ops.get_phase_distribution(self.levelset)

    def get_viscosity(self, physics_config) -> ScalarField:
        """粘性場を計算

        Args:
            physics_config: 物理設定

        Returns:
            粘性場
        """
        # 相の物性値から粘性場を計算
        phases = physics_config.phases
        return self._interface_ops.get_property_field(
            self.levelset, value1=phases[0].viscosity, value2=phases[1].viscosity
        )

    def get_density(self, physics_config) -> ScalarField:
        """密度場を計算

        Args:
            physics_config: 物理設定

        Returns:
            密度場
        """
        # 相の物性値から密度場を計算
        phases = physics_config.phases
        return self._interface_ops.get_property_field(
            self.levelset, value1=phases[0].density, value2=phases[1].density
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        # 界面の診断情報を取得
        interface_diagnostics = self._interface_ops.get_diagnostics(self.levelset)

        return {
            "time": self.time,
            "velocity_max": float(
                np.max([np.abs(v.data).max() for v in self.velocity.components])
            ),
            "pressure_max": float(np.abs(self.pressure.data).max()),
            "levelset_min": float(self.levelset.data.min()),
            "levelset_max": float(self.levelset.data.max()),
            "interface_geometry": interface_diagnostics,
            **self.diagnostics,
        }

    def copy(self) -> "SimulationState":
        """状態の深いコピーを作成

        Returns:
            コピーされた状態
        """
        return SimulationState(
            time=self.time,
            velocity=self.velocity.copy(),
            levelset=ScalarField(
                self.levelset.shape, self.levelset.dx, self.levelset.data.copy()
            ),
            pressure=self.pressure.copy(),
            diagnostics=self.diagnostics.copy() if self.diagnostics else None,
        )

    def update(self, derivative: "SimulationState", dt: float) -> None:
        """状態を更新

        Args:
            derivative: 時間微分
            dt: 時間刻み幅
        """
        self.time += dt
        for i, comp in enumerate(self.velocity.components):
            comp.data += dt * derivative.velocity.components[i].data
        self.levelset.data += dt * derivative.levelset.data
        self.pressure.data += dt * derivative.pressure.data

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

        # 界面関数の再構築
        levelset = ScalarField(data["levelset_data"].shape)
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
