"""物理パラメータの設定を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from .base import BaseConfig


@dataclass
class PhaseConfig(BaseConfig):
    """流体の物性値を保持するクラス"""

    phase: str  # 相の名前
    density: float = 1000.0  # 密度 [kg/m³]
    viscosity: float = 1.0e-3  # 動粘性係数 [Pa·s]
    surface_tension: float = 0.0  # 表面張力係数 [N/m]

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if not self.phase:
            raise ValueError("相の名前は必須です")
        if self.density <= 0:
            raise ValueError("密度は正の値である必要があります")
        if self.viscosity <= 0:
            raise ValueError("粘性は正の値である必要があります")
        if self.surface_tension < 0:
            raise ValueError("表面張力は非負である必要があります")

    def load(self, config_dict: Dict[str, Any]) -> "PhaseConfig":
        """辞書から設定を読み込む"""
        # phaseは必須なので、渡されていない場合は現在の値を使用
        phase = config_dict.get("phase", self.phase)
        return PhaseConfig(
            phase=phase,
            density=config_dict.get("density", self.density),
            viscosity=config_dict.get("viscosity", self.viscosity),
            surface_tension=config_dict.get("surface_tension", self.surface_tension),
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {
            "phase": self.phase,
            "density": self.density,
            "viscosity": self.viscosity,
            "surface_tension": self.surface_tension,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PhaseConfig":
        """辞書から設定を復元"""
        if "phase" not in config_dict:
            raise ValueError("相の名前(phase)は必須です")
        return cls(
            phase=config_dict["phase"],
            density=config_dict.get("density", 1000.0),
            viscosity=config_dict.get("viscosity", 1.0e-3),
            surface_tension=config_dict.get("surface_tension", 0.0),
        )


@dataclass
class PhysicsConfig(BaseConfig):
    """物理パラメータの設定を保持するクラス"""

    gravity: float = 9.81  # 重力加速度 [m/s²]
    surface_tension: float = 0.072  # 表面張力係数 [N/m]
    phases: List[PhaseConfig] = field(default_factory=list)  # 相の設定リスト

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.gravity <= 0:
            raise ValueError("重力加速度は正の値である必要があります")
        if self.surface_tension < 0:
            raise ValueError("表面張力係数は非負である必要があります")
        for phase in self.phases:
            phase.validate()

    def load(self, config_dict: Dict[str, Any]) -> "PhysicsConfig":
        """辞書から設定を読み込む"""
        phases = []
        for phase_dict in config_dict.get("phases", []):
            phase = PhaseConfig.from_dict(phase_dict)
            phases.append(phase)

        return PhysicsConfig(
            gravity=config_dict.get("gravity", self.gravity),
            surface_tension=config_dict.get("surface_tension", self.surface_tension),
            phases=phases,
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {
            "gravity": self.gravity,
            "surface_tension": self.surface_tension,
            "phases": [phase.to_dict() for phase in self.phases],
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PhysicsConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)

    def get_phase_by_name(self, name: str) -> PhaseConfig:
        """名前で相の設定を取得"""
        for phase in self.phases:
            if phase.phase.lower() == name.lower():
                return phase
        raise ValueError(f"指定された相が見つかりません: {name}")


@dataclass
class DomainConfig(BaseConfig):
    """計算領域の設定を保持するクラス"""

    dimensions: List[int] = field(
        default_factory=lambda: [32, 32, 32]
    )  # [nx, ny, nz]の格子点数
    size: List[float] = field(
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
