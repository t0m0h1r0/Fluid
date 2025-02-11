"""設定のインターフェースと抽象基底クラス"""

from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Generic


T = TypeVar("T")


class ConfigValidator(ABC, Generic[T]):
    """設定検証のための抽象基底クラス"""

    @abstractmethod
    def validate(self) -> None:
        """設定値の妥当性を検証"""
        pass


class ConfigLoader(ABC, Generic[T]):
    """設定読み込みのための抽象基底クラス"""

    @abstractmethod
    def load(self, config_dict: Dict[str, Any]) -> T:
        """辞書から設定を読み込む"""
        pass


class ConfigSerializer(ABC, Generic[T]):
    """設定のシリアライズ/デシリアライズのインターフェース"""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> T:
        """辞書から設定を復元"""
        pass
