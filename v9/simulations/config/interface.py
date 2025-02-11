"""界面の設定を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .base import BaseConfig, Phase, load_config_safely


@dataclass
class InterfaceConfig(BaseConfig):
    """界面の設定を保持するクラス"""

    phase: Phase = Phase.WATER
    object_type: str = "background"  # "background", "layer", "sphere"
    height: Optional[float] = None  # レイヤー用
    center: Optional[List[float]] = None  # 球体用
    radius: Optional[float] = None  # 球体用

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
        # デフォルト値を設定しつつ、入力された値で上書き
        merged_config = load_config_safely(
            config_dict,
            {
                "phase": "water",
                "type": "background",
                "height": None,
                "center": None,
                "radius": None,
            },
        )

        return InterfaceConfig(
            phase=Phase[merged_config.get("phase", "water").upper()],
            object_type=merged_config.get("type", "background"),
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


@dataclass
class InitialConditionConfig(BaseConfig):
    """初期条件の設定を保持するクラス"""

    background: Dict[str, str] = field(default_factory=lambda: {"phase": "nitrogen"})
    objects: List[Dict[str, Any]] = field(default_factory=list)
    velocity: Dict[str, str] = field(default_factory=lambda: {"type": "zero"})

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        # 背景相のバリデーション
        if "phase" not in self.background:
            raise ValueError("背景相には相の指定が必要です")

        # オブジェクトのバリデーション
        for obj in self.objects:
            InterfaceConfig(**obj).validate()

        # 速度場のバリデーション
        if "type" not in self.velocity:
            raise ValueError("初期速度場の種類の指定が必要です")

    def load(self, config_dict: Dict[str, Any]) -> "InitialConditionConfig":
        """辞書から設定を読み込む"""
        # デフォルト値を設定しつつ、入力された値で上書き
        merged_config = load_config_safely(
            config_dict,
            {
                "background": {"phase": "nitrogen"},
                "objects": [],
                "velocity": {"type": "zero"},
            },
        )

        return InitialConditionConfig(
            background=merged_config.get("background", {"phase": "nitrogen"}),
            objects=[
                {k: str(v) for k, v in obj.items()}
                for obj in merged_config.get("objects", [])
            ],
            velocity=merged_config.get("velocity", {"type": "zero"}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {
            "background": self.background,
            "objects": self.objects,
            "velocity": self.velocity,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "InitialConditionConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)
