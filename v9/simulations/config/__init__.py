"""設定管理パッケージの初期化"""

from .base import BaseConfig, Phase, BoundaryType, load_config_safely
from .interfaces import ConfigValidator, ConfigLoader, ConfigSerializer
from .physics import PhysicsConfig, DomainConfig
from .numerical import NumericalConfig
from .boundary import BoundaryConfig
from .interface import InterfaceConfig, InitialConditionConfig
from .output import OutputConfig
from .simulation import SimulationConfig

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
    "PhysicsConfig",
    "DomainConfig",
    "NumericalConfig",
    "BoundaryConfig",
    "InterfaceConfig",
    "InitialConditionConfig",
    "OutputConfig",
    # ユーティリティ関数
    "load_config_safely",
]
