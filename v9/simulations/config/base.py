"""設定の基本的な列挙型と基底クラス"""

from enum import Enum, auto
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from .interfaces import ConfigValidator, ConfigLoader, ConfigSerializer


class Phase(Enum):
    """流体の相を表す列挙型"""

    WATER = auto()
    NITROGEN = auto()
    GAS = auto()
    LIQUID = auto()


class BoundaryType(Enum):
    """境界条件の種類を表す列挙型"""

    PERIODIC = "periodic"
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"


@dataclass
class BaseConfig(ConfigValidator, ConfigLoader, ConfigSerializer):
    """設定の基底クラス"""

    def validate(self) -> None:
        """デフォルトの検証メソッド（サブクラスでオーバーライド）"""
        pass

    def load(self, config_dict: Dict[str, Any]) -> "BaseConfig":
        """デフォルトの読み込みメソッド（サブクラスでオーバーライド）"""
        raise NotImplementedError("サブクラスで実装する必要があります")

    def to_dict(self) -> Dict[str, Any]:
        """オブジェクトを辞書に変換"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """辞書からオブジェクトを生成（サブクラスでオーバーライド）"""
        raise NotImplementedError("サブクラスで実装する必要があります")

    def __getitem__(self, key: str) -> Any:
        """辞書風のインデックスアクセスを可能にする"""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(
                f"'{self.__class__.__name__}' オブジェクトに '{key}' は存在しません"
            )

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """辞書風のgetメソッドを実装"""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __contains__(self, key: str) -> bool:
        """in演算子をサポート"""
        return hasattr(self, key)


def load_config_safely(
    config_dict: Dict[str, Any], default_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    設定辞書を安全に読み込む

    Args:
        config_dict: 読み込む設定辞書
        default_dict: デフォルト値の辞書

    Returns:
        マージされた設定辞書
    """
    # デフォルト値の設定
    if default_dict is None:
        default_dict = {}

    # 再帰的にデフォルト値をマージ
    def deep_merge(default, override):
        if isinstance(default, dict) and isinstance(override, dict):
            merged = default.copy()
            for key, value in override.items():
                merged[key] = deep_merge(merged.get(key, {}), value)
            return merged
        return override

    return deep_merge(default_dict, config_dict)
