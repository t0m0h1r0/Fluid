"""シミュレーション状態を保持するモジュール"""

from dataclasses import dataclass
from typing import Optional

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.levelset.properties import PropertiesManager
import numpy as np


@dataclass
class SimulationState:
    """シミュレーションの状態を保持するクラス

    流れ場と界面の情報を保持し、シミュレーション全体の
    状態管理を担当します。
    """

    velocity: VectorField
    levelset: LevelSetField
    pressure: ScalarField
    time: float = 0.0
    properties: Optional[PropertiesManager] = None

    def copy(self) -> "SimulationState":
        """状態の深いコピーを作成

        Returns:
            コピーされた状態
        """
        return SimulationState(
            velocity=self.velocity.copy(),
            levelset=self.levelset.copy(),
            pressure=self.pressure.copy(),
            time=self.time,
            properties=self.properties,  # PropertiesManagerは共有して問題ない
        )

    def get_phase_fields(self) -> dict:
        """各相の特性を表すフィールドを取得

        Returns:
            フィールドの辞書
        """
        if self.properties is None:
            return {}

        # 密度場と粘性係数場を取得
        density = self.properties.get_density(self.levelset)
        viscosity = self.properties.get_viscosity(self.levelset)

        return {
            "density": density,
            "viscosity": viscosity,
        }

    def get_energy(self) -> dict:
        """エネルギー関連の量を計算

        Returns:
            エネルギー関連の値の辞書
        """
        # 運動エネルギー
        kinetic_energy = 0.0
        for component in self.velocity.components:
            kinetic_energy += 0.5 * np.sum(component.data**2)

        # 位置エネルギー（重力による）
        if self.properties is not None:
            density = self.properties.get_density(self.levelset)
            g = 9.81  # 重力加速度
            z = np.linspace(0, 1, self.levelset.shape[2])
            Z = np.tile(z, (self.levelset.shape[0], self.levelset.shape[1], 1))
            potential_energy = np.sum(density * g * Z)
        else:
            potential_energy = 0.0

        return {
            "kinetic": float(kinetic_energy),
            "potential": float(potential_energy),
            "total": float(kinetic_energy + potential_energy),
        }
