"""出力設定を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Union
from pathlib import Path
from .base import BaseConfig, load_config_safely


@dataclass
class OutputConfig(BaseConfig):
    """出力の設定を保持するクラス"""

    output_dir: Union[str, Path] = Path("results/visualization")
    format: str = "png"
    dpi: int = 300
    colormap: str = "viridis"
    show_colorbar: bool = True
    show_axes: bool = True
    show_grid: bool = False
    slices: Dict[str, List[Union[str, float]]] = field(
        default_factory=lambda: {"axes": ["xy", "xz", "yz"], "positions": [0.5]}
    )
    fields: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "velocity": {
                "enabled": True,
                "plot_types": ["vector", "magnitude"],
                "scale": 1.0,
                "density": 20,
                "color": "black",
                "alpha": 0.7,
            },
            "pressure": {
                "enabled": True,
                "plot_types": ["scalar", "contour"],
                "levels": 20,
                "alpha": 0.5,
            },
            "levelset": {
                "enabled": True,
                "plot_types": ["interface", "contour"],
                "levels": [0],
                "colors": ["black"],
                "linewidth": 2.0,
            },
        }
    )

    def __post_init__(self):
        """初期化後の処理"""
        # パスをPath型に変換
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # 出力ディレクトリの作成（存在しない場合）
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        # 出力ディレクトリのバリデーション
        if not self.output_dir:
            raise ValueError("出力ディレクトリは空にできません")

        # 解像度のバリデーション
        if self.dpi <= 0:
            raise ValueError("dpiは正の値である必要があります")

        # スライス軸のバリデーション
        valid_axes = {"xy", "xz", "yz"}
        if not all(axis in valid_axes for axis in self.slices.get("axes", [])):
            raise ValueError(f"無効なスライス軸。有効な値: {valid_axes}")

        # スライス位置のバリデーション
        if not all(0 <= pos <= 1 for pos in self.slices.get("positions", [])):
            raise ValueError("スライス位置は0から1の間である必要があります")

    def load(self, config_dict: Dict[str, Any]) -> "OutputConfig":
        """辞書から設定を読み込む"""
        # デフォルト値を設定しつつ、入力された値で上書き
        merged_config = load_config_safely(
            config_dict,
            {
                "output_dir": "results/visualization",
                "format": "png",
                "dpi": 300,
                "colormap": "viridis",
                "show_colorbar": True,
                "show_axes": True,
                "show_grid": False,
                "slices": {"axes": ["xy", "xz", "yz"], "positions": [0.5]},
                "fields": {
                    "velocity": {
                        "enabled": True,
                        "plot_types": ["vector", "magnitude"],
                        "scale": 1.0,
                        "density": 20,
                        "color": "black",
                        "alpha": 0.7,
                    },
                    "pressure": {
                        "enabled": True,
                        "plot_types": ["scalar", "contour"],
                        "levels": 20,
                        "alpha": 0.5,
                    },
                    "levelset": {
                        "enabled": True,
                        "plot_types": ["interface", "contour"],
                        "levels": [0],
                        "colors": ["black"],
                        "linewidth": 2.0,
                    },
                },
            },
        )

        return OutputConfig(
            output_dir=Path(merged_config.get("output_dir", "results/visualization")),
            format=merged_config.get("format", "png"),
            dpi=merged_config.get("dpi", 300),
            colormap=merged_config.get("colormap", "viridis"),
            show_colorbar=merged_config.get("show_colorbar", True),
            show_axes=merged_config.get("show_axes", True),
            show_grid=merged_config.get("show_grid", False),
            slices=merged_config.get(
                "slices", {"axes": ["xy", "xz", "yz"], "positions": [0.5]}
            ),
            fields=merged_config.get(
                "fields",
                {
                    "velocity": {
                        "enabled": True,
                        "plot_types": ["vector", "magnitude"],
                        "scale": 1.0,
                        "density": 20,
                        "color": "black",
                        "alpha": 0.7,
                    },
                    "pressure": {
                        "enabled": True,
                        "plot_types": ["scalar", "contour"],
                        "levels": 20,
                        "alpha": 0.5,
                    },
                    "levelset": {
                        "enabled": True,
                        "plot_types": ["interface", "contour"],
                        "levels": [0],
                        "colors": ["black"],
                        "linewidth": 2.0,
                    },
                },
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {
            "output_dir": str(self.output_dir),
            "format": self.format,
            "dpi": self.dpi,
            "colormap": self.colormap,
            "show_colorbar": self.show_colorbar,
            "show_axes": self.show_axes,
            "show_grid": self.show_grid,
            "slices": self.slices,
            "fields": self.fields,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OutputConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)
