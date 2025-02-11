"""界面の設定を管理するモジュール"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from .base import BaseConfig, Phase, load_config_safely


@dataclass
class InterfaceConfig(BaseConfig):
    """界面の設定を保持するクラス"""

    phase: Phase = Phase.WATER
    object_type: str = "background"  # "background", "layer", "sphere"
    height: Optional[float] = None  # レイヤー用
    center: Optional[List[float]] = None  # 球体用
    radius: Optional[float] = None  # 球体用

    def __init__(
        self,
        phase: Optional[Union[str, Phase]] = None,
        object_type: Optional[str] = None,
        type: Optional[str] = None,  # 追加
        height: Optional[float] = None,
        center: Optional[List[float]] = None,
        radius: Optional[float] = None,
    ):
        """インターフェース設定を初期化

        `type` キーワードを `object_type` として処理
        """
        if type is not None and object_type is None:
            object_type = type

        # Phase の変換
        if isinstance(phase, str):
            phase = Phase[phase.upper()]

        # デフォルト値の設定
        self.phase = phase or Phase.WATER
        self.object_type = object_type or "background"
        self.height = height
        self.center = center
        self.radius = radius

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.object_type == "background":
            if any([self.height, self.center, self.radius]):
                raise ValueError("背景相には高さ、中心、半径は指定できません")
        elif self.object_type == "layer":
            if self.height is not None and not 0 <= self.height <= 1:
                raise ValueError("高さは0から1の間である必要があります")
            if any([self.center, self.radius]):
                raise ValueError("レイヤーには高さのみ指定してください")
        elif self.object_type == "sphere":
            if not self.center or len(self.center) != 3:
                raise ValueError("球体には3次元の中心座標が必要です")
            if not self.radius or self.radius <= 0:
                raise ValueError("球体には正の半径が必要です")
            if self.height is not None:
                raise ValueError("球体には高さは指定できません")
        else:
            raise ValueError(f"未対応のオブジェクトタイプ: {self.object_type}")

    def load(self, config_dict: Dict[str, Any]) -> "InterfaceConfig":
        """辞書から設定を読み込む"""
        # 'type' キーを 'object_type' に変換
        if "type" in config_dict:
            config_dict["object_type"] = config_dict.pop("type")

        # デフォルト値を設定しつつ、入力された値で上書き
        merged_config = load_config_safely(
            config_dict,
            {
                "phase": "water",
                "object_type": "background",
                "height": None,
                "center": None,
                "radius": None,
            },
        )

        return InterfaceConfig(
            phase=Phase[merged_config.get("phase", "water").upper()],
            object_type=merged_config.get("object_type", "background"),
            height=merged_config.get("height"),
            center=merged_config.get("center"),
            radius=merged_config.get("radius"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {
            "phase": self.phase.name.lower(),
            "type": self.object_type,
            "height": self.height,
            "center": self.center,
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "InterfaceConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)
