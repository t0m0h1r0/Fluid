"""シミュレーション状態を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path
import numpy as np
import json

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager


@dataclass
class SimulationState:
    """シミュレーションの状態を管理するクラス"""

    velocity: VectorField
    pressure: ScalarField
    levelset: LevelSetField
    properties: PropertiesManager
    time: float = 0.0
    iteration: int = 0
    next_save: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def update_statistics(self):
        """シミュレーション統計を更新"""
        # 速度場の統計
        vel_mag = self.velocity.magnitude()
        self.statistics.update(
            {
                "max_velocity": np.max(vel_mag.data),
                "mean_velocity": np.mean(vel_mag.data),
                "max_pressure": np.max(np.abs(self.pressure.data)),
                "kinetic_energy": 0.5
                * np.sum(vel_mag.data**2)
                * self.velocity.dx**self.velocity.ndim,
                "interface_area": self.levelset.compute_area(),
                "phase1_volume": np.sum(self.levelset.heaviside())
                * self.velocity.dx**self.velocity.ndim,
            }
        )

        # 発散チェック
        div = self.velocity.divergence()
        self.statistics["max_divergence"] = np.max(np.abs(div.data))

    def save(self, directory: Path):
        """状態をファイルに保存"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # フィールドデータの保存
        np.save(directory / "velocity.npy", [v.data for v in self.velocity.components])
        np.save(directory / "pressure.npy", self.pressure.data)
        np.save(directory / "levelset.npy", self.levelset.data)

        # メタデータの保存
        meta = {
            "time": self.time,
            "iteration": self.iteration,
            "next_save": self.next_save,
            "statistics": self.statistics,
            "meta": self.meta,
        }
        with (directory / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: Path, config: Dict[str, Any]) -> "SimulationState":
        """保存された状態を読み込み"""
        directory = Path(directory)

        # メタデータの読み込み
        with (directory / "meta.json").open("r") as f:
            meta = json.load(f)

        # グリッドサイズの取得
        velocity_data = np.load(directory / "velocity.npy")
        shape = velocity_data[0].shape
        dx = config["domain"]["size"][0] / shape[0]

        # フィールドの再構築
        velocity = VectorField(shape, dx)
        for i, data in enumerate(velocity_data):
            velocity.components[i].data = data

        pressure = ScalarField(shape, dx)
        pressure.data = np.load(directory / "pressure.npy")

        levelset = LevelSetField(shape, dx)
        levelset.data = np.load(directory / "levelset.npy")

        # 物性値マネージャーの再構築
        properties = PropertiesManager(
            phase1=config["physics"]["phases"]["water"],
            phase2=config["physics"]["phases"]["air"],
        )

        return cls(
            velocity=velocity,
            pressure=pressure,
            levelset=levelset,
            properties=properties,
            time=meta["time"],
            iteration=meta["iteration"],
            next_save=meta["next_save"],
            statistics=meta["statistics"],
            meta=meta["meta"],
        )

    def is_valid(self) -> bool:
        """状態の妥当性をチェック"""
        # NaNチェック
        for v in self.velocity.components:
            if np.any(np.isnan(v.data)):
                return False
        if np.any(np.isnan(self.pressure.data)):
            return False
        if np.any(np.isnan(self.levelset.data)):
            return False

        # 発散チェック
        if self.statistics.get("max_divergence", float("inf")) > 1e-3:
            return False

        return True
