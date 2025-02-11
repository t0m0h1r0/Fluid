"""設定管理パッケージの初期化"""

from .base import BaseConfig, Phase, BoundaryType, load_config_safely
from .interfaces import ConfigValidator, ConfigLoader, ConfigSerializer
from .physics import PhaseConfig, DomainConfig
from .numerical import NumericalConfig
from .boundary import BoundaryConfig
from .interface import InterfaceConfig, InitialConditionConfig
from .output import OutputConfig

# 設定を統合するメインクラス
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class SimulationConfig(BaseConfig):
    """シミュレーション全体の設定を保持するクラス"""

    domain: DomainConfig = field(default_factory=DomainConfig)
    phases: Dict[str, PhaseConfig] = field(default_factory=dict)
    boundary_conditions: BoundaryConfig = field(default_factory=BoundaryConfig)
    initial_conditions: InitialConditionConfig = field(
        default_factory=InitialConditionConfig
    )
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    interfaces: list[InterfaceConfig] = field(default_factory=list)

    def validate(self) -> None:
        """全体の設定値の妥当性を検証"""
        self.domain.validate()

        for phase in self.phases.values():
            phase.validate()

        self.boundary_conditions.validate()
        self.initial_conditions.validate()
        self.numerical.validate()
        self.output.validate()

        for interface in self.interfaces:
            interface.validate()

    def load(self, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """辞書から設定を読み込む"""
        # デフォルト値を設定しつつ、入力された値で上書き
        merged_config = load_config_safely(config_dict, {})

        return SimulationConfig(
            domain=DomainConfig.from_dict(merged_config.get("domain", {})),
            phases={
                name: PhaseConfig.from_dict(props)
                for name, props in merged_config.get("phases", {}).items()
            },
            boundary_conditions=BoundaryConfig.from_dict(
                merged_config.get("boundary_conditions", {})
            ),
            initial_conditions=InitialConditionConfig.from_dict(
                merged_config.get("initial_conditions", {})
            ),
            numerical=NumericalConfig.from_dict(merged_config.get("numerical", {})),
            output=OutputConfig.from_dict(merged_config.get("output", {})),
            interfaces=[
                InterfaceConfig.from_dict(obj)
                for obj in merged_config.get("interfaces", [])
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {
            "domain": self.domain.to_dict(),
            "phases": {name: phase.to_dict() for name, phase in self.phases.items()},
            "boundary_conditions": self.boundary_conditions.to_dict(),
            "initial_conditions": self.initial_conditions.to_dict(),
            "numerical": self.numerical.to_dict(),
            "output": self.output.to_dict(),
            "interfaces": [interface.to_dict() for interface in self.interfaces],
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SimulationConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)

    @classmethod
    def from_yaml(cls, filepath: str) -> "SimulationConfig":
        """YAMLファイルから設定を読み込む"""
        import yaml

        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


# エクスポートする公開インターフェース
__all__ = [
    # 基本的な設定クラス
    "SimulationConfig",
    # 基底クラスとインターフェース
    "BaseConfig",
    "ConfigValidator",
    "ConfigLoader",
    "ConfigSerializer",
    # 列挙型
    "Phase",
    "BoundaryType",
    # 個別の設定クラス
    "PhaseConfig",
    "DomainConfig",
    "NumericalConfig",
    "BoundaryConfig",
    "InterfaceConfig",
    "InitialConditionConfig",
    "OutputConfig",
    # ユーティリティ関数
    "load_config_safely",
]
