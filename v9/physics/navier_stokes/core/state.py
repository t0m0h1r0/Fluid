"""Navier-Stokes方程式の状態を管理するモジュール

このモジュールは、Navier-Stokes方程式の状態（速度場、圧力場など）を
管理するためのクラスを提供します。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.levelset.properties import PropertiesManager


@dataclass
class NavierStokesState:
    """Navier-Stokes方程式の状態を表すクラス

    速度場、圧力場、レベルセット場などの物理量を保持します。
    """

    velocity: VectorField
    pressure: ScalarField
    levelset: Optional[LevelSetField] = None
    properties: Optional[PropertiesManager] = None
    time: float = 0.0

    def copy(self) -> "NavierStokesState":
        """状態の深いコピーを作成"""
        return NavierStokesState(
            velocity=self.velocity.copy(),
            pressure=self.pressure.copy(),
            levelset=self.levelset.copy() if self.levelset is not None else None,
            properties=self.properties,  # PropertiesManagerは共有して問題ない
            time=self.time,
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """状態の診断情報を取得"""
        diag = {
            "time": self.time,
            "velocity": {
                "max": float(
                    max(np.max(np.abs(c.data)) for c in self.velocity.components)
                ),
                "energy": float(
                    sum(np.sum(c.data**2) for c in self.velocity.components)
                    * 0.5
                    * self.velocity.dx**3
                ),
            },
            "pressure": {
                "min": float(np.min(self.pressure.data)),
                "max": float(np.max(self.pressure.data)),
            },
        }

        if self.levelset is not None:
            diag["levelset"] = self.levelset.get_diagnostics()

        return diag

    def save_state(self) -> Dict[str, Any]:
        """状態を保存用の辞書として取得"""
        state_dict = {
            "velocity": self.velocity.save_state(),
            "pressure": self.pressure.save_state(),
            "time": self.time,
        }

        if self.levelset is not None:
            state_dict["levelset"] = self.levelset.save_state()

        return state_dict

    @classmethod
    def load_state(
        cls,
        state_dict: Dict[str, Any],
        shape: tuple,
        dx: float,
        properties: Optional[PropertiesManager] = None,
    ) -> "NavierStokesState":
        """保存された状態から復元

        Args:
            state_dict: 保存された状態の辞書
            shape: グリッドの形状
            dx: グリッド間隔
            properties: 物性値マネージャー（オプション）

        Returns:
            復元された状態
        """
        # 速度場の復元
        velocity = VectorField(shape, dx)
        velocity.load_state(state_dict["velocity"])

        # 圧力場の復元
        pressure = ScalarField(shape, dx)
        pressure.load_state(state_dict["pressure"])

        # レベルセット場の復元（存在する場合）
        levelset = None
        if "levelset" in state_dict:
            levelset = LevelSetField(shape, dx)
            levelset.load_state(state_dict["levelset"])

        return cls(
            velocity=velocity,
            pressure=pressure,
            levelset=levelset,
            properties=properties,
            time=float(state_dict["time"]),
        )

    def validate(self) -> bool:
        """状態の妥当性を検証

        Returns:
            状態が有効であればTrue
        """
        # 基本的な形状の一貫性チェック
        shape = self.velocity.shape
        if self.pressure.shape != shape:
            return False
        if self.levelset is not None and self.levelset.shape != shape:
            return False

        # グリッド間隔の一貫性チェック
        dx = self.velocity.dx
        if abs(self.pressure.dx - dx) > 1e-10:
            return False
        if self.levelset is not None and abs(self.levelset.dx - dx) > 1e-10:
            return False

        # 物理的な妥当性チェック
        if np.any(np.isnan(self.pressure.data)) or np.any(np.isinf(self.pressure.data)):
            return False
        for comp in self.velocity.components:
            if np.any(np.isnan(comp.data)) or np.any(np.isinf(comp.data)):
                return False

        return True

    def __str__(self) -> str:
        """文字列表現"""
        diag = self.get_diagnostics()
        return (
            f"NavierStokesState at t={self.time:.3f}:\n"
            f"  Velocity: max={diag['velocity']['max']:.3e}, "
            f"energy={diag['velocity']['energy']:.3e}\n"
            f"  Pressure: [{diag['pressure']['min']:.3e}, "
            f"{diag['pressure']['max']:.3e}]"
        )
