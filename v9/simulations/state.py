"""シミュレーション状態を保持するモジュール

リファクタリングされたphysics/パッケージに対応した更新版
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from physics.levelset import LevelSetField, LevelSetPropertiesManager
from core.field import VectorField, ScalarField, ConservedField


@dataclass
class SimulationState:
    """シミュレーションの状態を保持するクラス

    流れ場と界面の情報を保持し、高度な状態管理を実現
    """

    velocity: VectorField
    levelset: LevelSetField
    pressure: ScalarField
    time: float = 0.0
    properties: Optional[LevelSetPropertiesManager] = None

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
            properties=self.properties
        )

    def get_phase_fields(self) -> dict:
        """各相の特性を表すフィールドを取得

        Returns:
            フィールドの辞書
        """
        if self.properties is None:
            return {}

        return {
            "density": self.properties.compute_density(self.levelset),
            "viscosity": self.properties.compute_viscosity(self.levelset),
        }

    def get_energy(self) -> dict:
        """エネルギー関連の量を計算

        Returns:
            エネルギー関連の値の辞書
        """
        # 運動エネルギー
        kinetic_energy = sum(
            0.5 * np.sum(comp.data**2) 
            for comp in self.velocity.components
        )

        # 位置エネルギー（重力による）
        potential_energy = 0.0
        if self.properties is not None:
            density = self.properties.compute_density(self.levelset)
            g = 9.81  # 重力加速度
            z = np.linspace(0, 1, self.levelset.shape[2])
            potential_energy = np.sum(density * g * z)

        return {
            "kinetic": float(kinetic_energy),
            "potential": float(potential_energy),
            "total": float(kinetic_energy + potential_energy),
        }

    def compute_conservation(self) -> dict:
        """保存則の評価

        Returns:
            各物理量の保存則評価結果
        """
        conserved_vars = {
            "mass": ConservedField(self.properties.compute_density(self.levelset)),
            "momentum": ConservedField(
                sum(v.data for v in self.velocity.components)
            )
        }

        return {
            var: field.conservation_error 
            for var, field in conserved_vars.items()
        }