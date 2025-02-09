"""シミュレーション状態を保持するモジュール"""

from dataclasses import dataclass
from typing import Optional

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager


@dataclass
class SimulationState:
    """シミュレーションの状態を保持するクラス"""

    velocity: VectorField
    levelset: LevelSetField
    pressure: ScalarField
    time: float = 0.0
    properties: Optional[PropertiesManager] = None

    def copy(self) -> "SimulationState":
        """状態の深いコピーを作成"""
        return SimulationState(
            velocity=self.velocity.copy(),
            levelset=self.levelset.copy(),
            pressure=self.pressure.copy(),
            time=self.time,
            properties=self.properties,  # PropertiesManagerは共有して問題ない
        )
