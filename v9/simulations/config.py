"""シミュレーション設定を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from physics.levelset.initializer import Phase


@dataclass
class PhaseConfig:
    """流体の物性値を保持するクラス"""

    density: float
    viscosity: float
    surface_tension: float = 0.0


@dataclass
class DomainConfig:
    """計算領域の設定を保持するクラス"""

    dimensions: Dict[str, int]  # 'x', 'y', 'z'の格子点数
    size: Dict[str, float]  # 'x', 'y', 'z'の物理サイズ

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if not all(dim > 0 for dim in self.dimensions.values()):
            raise ValueError("格子点数は正の値である必要があります")
        if not all(size > 0 for size in self.size.values()):
            raise ValueError("領域サイズは正の値である必要があります")


@dataclass
class InterfaceConfig:
    """界面の設定を保持するクラス"""

    phase: Phase
    object_type: str  # "background", "layer", "sphere"
    height: Optional[float] = None  # レイヤー用
    center: Optional[Tuple[float, float, float]] = None  # 球体用
    radius: Optional[float] = None  # 球体用

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.object_type == "background":
            if any([self.height, self.center, self.radius]):
                raise ValueError("背景設定には追加パラメータは不要です")

        elif self.object_type == "layer":
            if not self.height or self.height <= 0:
                raise ValueError("レイヤーには正の高さが必要です")
            if any([self.center, self.radius]):
                raise ValueError("レイヤーには高さのみ指定してください")

        elif self.object_type == "sphere":
            if not self.center or not self.radius or self.radius <= 0:
                raise ValueError("球には中心座標と正の半径が必要です")
            if self.height is not None:
                raise ValueError("球には不要なパラメータが指定されています")


@dataclass
class SimulationConfig:
    """シミュレーション全体の設定を保持するクラス"""

    domain: DomainConfig
    phases: Dict[Phase, PhaseConfig]
    interfaces: List[InterfaceConfig] = field(default_factory=list)
    output_dir: str = "results"

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        # ドメインの検証
        self.domain.validate()

        # 各界面設定の検証
        for interface in self.interfaces:
            interface.validate()
            if interface.phase not in self.phases:
                raise ValueError(f"未定義の相です: {interface.phase}")

        # 背景設定の確認
        background_objects = [
            obj for obj in self.interfaces if obj.object_type == "background"
        ]
        if not background_objects:
            raise ValueError("背景相の設定が必要です")
        if len(background_objects) > 1:
            raise ValueError("背景相の設定は1つのみ可能です")

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SimulationConfig":
        """辞書から設定を生成"""
        domain = DomainConfig(**config_dict["domain"])

        # 流体の物性値を設定
        phases = {
            Phase[name.upper()]: PhaseConfig(**props)
            for name, props in config_dict["phases"].items()
        }

        # 界面の設定を生成
        interfaces = []
        for obj_dict in config_dict.get("interfaces", []):
            interface_config = InterfaceConfig(
                phase=Phase[obj_dict["phase"].upper()],
                object_type=obj_dict["type"],
                height=obj_dict.get("height"),
                center=tuple(obj_dict.get("center", []))
                if "center" in obj_dict
                else None,
                radius=obj_dict.get("radius"),
            )
            interfaces.append(interface_config)

        return cls(
            domain=domain,
            phases=phases,
            interfaces=interfaces,
            output_dir=config_dict.get("output_dir", "results"),
        )
