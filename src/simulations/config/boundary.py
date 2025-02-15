"""境界条件の設定を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, Any
from .base import BaseConfig, BoundaryType, load_config_safely


@dataclass
class BoundaryConfig(BaseConfig):
    """境界条件の設定を保持するクラス"""

    x: Dict[str, str] = field(
        default_factory=lambda: {
            "left": BoundaryType.PERIODIC.value,
            "right": BoundaryType.PERIODIC.value,
        }
    )
    y: Dict[str, str] = field(
        default_factory=lambda: {
            "front": BoundaryType.PERIODIC.value,
            "back": BoundaryType.PERIODIC.value,
        }
    )
    z: Dict[str, str] = field(
        default_factory=lambda: {
            "bottom": BoundaryType.NEUMANN.value,
            "top": BoundaryType.NEUMANN.value,
        }
    )

    def validate(self) -> None:
        """境界条件の妥当性を検証"""
        valid_types = {bt.value for bt in BoundaryType}

        # 各方向の境界条件を検証
        for direction_name, direction in [("x", self.x), ("y", self.y), ("z", self.z)]:
            for side, boundary_type in direction.items():
                if boundary_type not in valid_types:
                    raise ValueError(
                        f"{direction_name}方向の{side}境界で無効な境界条件: {boundary_type}"
                    )

    def load(self, config_dict: Dict[str, Any]) -> "BoundaryConfig":
        """辞書から設定を読み込む"""
        # デフォルト値を設定しつつ、入力された値で上書き
        merged_config = load_config_safely(
            config_dict,
            {
                "x": {
                    "left": BoundaryType.PERIODIC.value,
                    "right": BoundaryType.PERIODIC.value,
                },
                "y": {
                    "front": BoundaryType.PERIODIC.value,
                    "back": BoundaryType.PERIODIC.value,
                },
                "z": {
                    "bottom": BoundaryType.NEUMANN.value,
                    "top": BoundaryType.NEUMANN.value,
                },
            },
        )

        return BoundaryConfig(
            x=merged_config.get("x", self.x),
            y=merged_config.get("y", self.y),
            z=merged_config.get("z", self.z),
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BoundaryConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)
