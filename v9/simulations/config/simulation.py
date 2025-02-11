"""シミュレーション全体の設定を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, Any
import yaml

from .base import BaseConfig, load_config_safely
from .physics import PhysicsConfig, DomainConfig
from .numerical import NumericalConfig
from .boundary import BoundaryConfig
from .interface import InterfaceConfig, InitialConditionConfig
from .output import OutputConfig


@dataclass
class SimulationConfig(BaseConfig):
    """シミュレーション全体の設定を保持するクラス"""

    domain: DomainConfig = field(default_factory=DomainConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
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
        self.physics.validate()
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
            physics=PhysicsConfig.from_dict(merged_config.get("physics", {})),
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
            "physics": self.physics.to_dict(),
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
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
