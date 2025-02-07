"""シミュレーション状態を管理するモジュール"""

from dataclasses import dataclass
from typing import Optional

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager


@dataclass
class SimulationState:
    """シミュレーションの状態を表すクラス"""

    velocity: VectorField
    pressure: ScalarField
    levelset: LevelSetField
    properties: Optional[PropertiesManager] = None

    def __post_init__(self):
        """初期化後の処理"""
        # components属性を追加
        self.components = self.velocity.components

    def update(self, **kwargs):
        """状態を更新

        Args:
            **kwargs: 更新する属性
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        # components属性を再設定
        self.components = self.velocity.components

    def copy(self):
        """状態の深いコピーを作成

        Returns:
            コピーされた状態
        """
        return SimulationState(
            velocity=self.velocity.copy(),
            pressure=self.pressure.copy(),
            levelset=self.levelset.copy(),
            properties=self.properties,
        )
