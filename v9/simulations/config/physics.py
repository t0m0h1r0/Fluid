"""物理パラメータの設定を管理するモジュール"""

from dataclasses import dataclass
from typing import Dict, Any
from .base import BaseConfig, Phase
from dataclasses import field


@dataclass
class PhaseConfig(BaseConfig):
    """流体の物性値を保持するクラス"""

    density: float = 1000.0  # 密度 [kg/m³]
    viscosity: float = 1.0e-3  # 動粘性係数 [Pa·s]
    surface_tension: float = 0.0  # 表面張力係数 [N/m]
    phase: Phase = Phase.WATER  # 相のタイプ

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.density <= 0:
            raise ValueError("密度は正の値である必要があります")
        if self.viscosity <= 0:
            raise ValueError("粘性は正の値である必要があります")
        if self.surface_tension < 0:
            raise ValueError("表面張力は非負である必要があります")

    def load(self, config_dict: Dict[str, Any]) -> "PhaseConfig":
        """辞書から設定を読み込む"""
        return PhaseConfig(
            density=config_dict.get("density", self.density),
            viscosity=config_dict.get("viscosity", self.viscosity),
            surface_tension=config_dict.get("surface_tension", self.surface_tension),
            phase=Phase[config_dict.get("phase", self.phase.name).upper()],
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {
            "density": self.density,
            "viscosity": self.viscosity,
            "surface_tension": self.surface_tension,
            "phase": self.phase.name.lower(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PhaseConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)


@dataclass
class DomainConfig(BaseConfig):
    """計算領域の設定を保持するクラス"""

    dimensions: list[int] = field(
        default_factory=lambda: [32, 32, 32]
    )  # [nx, ny, nz]の格子点数
    size: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0]
    )  # [Lx, Ly, Lz]の物理サイズ [m]

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if not all(isinstance(dim, int) and dim > 0 for dim in self.dimensions):
            raise ValueError("格子点数は正の整数である必要があります")
        if not all(isinstance(size, (int, float)) and size > 0 for size in self.size):
            raise ValueError("領域サイズは正の値である必要があります")
        if len(self.dimensions) != len(self.size):
            raise ValueError("dimensionsとsizeは同じ次元数である必要があります")
        if not (2 <= len(self.dimensions) <= 3):
            raise ValueError("2次元または3次元である必要があります")

    def load(self, config_dict: Dict[str, Any]) -> "DomainConfig":
        """辞書から設定を読み込む"""
        return DomainConfig(
            dimensions=config_dict.get("dimensions", self.dimensions),
            size=[float(s) for s in config_dict.get("size", self.size)],
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {"dimensions": self.dimensions, "size": self.size}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DomainConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)
