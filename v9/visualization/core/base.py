"""可視化システムの基本クラスとインターフェースを提供するモジュール

このモジュールは可視化システムの設定とインターフェースを定義します。すべての
可視化コンポーネントの基盤となります。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path


@dataclass
class ViewConfig:
    """可視化の表示設定

    3D表示やスライス表示の視点・断面を制御します。

    Attributes:
        elevation: 仰角（度）
        azimuth: 方位角（度）
        distance: 視点距離
        focal_point: 注視点座標 (x, y, z)
        slice_positions: 各軸でのスライス位置 [0-1]
        slice_axes: 表示する断面の軸 (例: ["xy", "yz", "xz"])
    """

    elevation: float = 30.0
    azimuth: float = 45.0
    distance: float = 10.0
    focal_point: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    slice_positions: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    slice_axes: List[str] = field(default_factory=lambda: ["xy", "yz", "xz"])

    def validate(self):
        """設定値の検証"""
        if len(self.slice_positions) != 3:
            raise ValueError("slice_positions must have 3 elements")
        if any(not 0 <= pos <= 1 for pos in self.slice_positions):
            raise ValueError("slice_positions must be between 0 and 1")
        valid_axes = {"xy", "yz", "xz", "yx", "zy", "zx"}
        if any(axis not in valid_axes for axis in self.slice_axes):
            raise ValueError(f"Invalid slice_axes. Must be one of {valid_axes}")


@dataclass
class VisualizationConfig:
    """可視化の基本設定

    出力形式や表示オプションを制御します。

    Attributes:
        output_dir: 出力ディレクトリ
        format: 出力フォーマット
        dpi: 解像度
        colormap: デフォルトのカラーマップ
        show_colorbar: カラーバーの表示
        show_axes: 軸の表示
        show_grid: グリッドの表示
        fields: フィールドごとの可視化設定
    """

    output_dir: Union[str, Path] = "results/visualization"
    format: str = "png"
    dpi: int = 300
    colormap: str = "viridis"
    show_colorbar: bool = True
    show_axes: bool = True
    show_grid: bool = False

    # フィールドごとの可視化設定
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
        """設定の後処理と検証"""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, name: str, timestamp: Optional[float] = None) -> Path:
        """出力ファイルパスを生成

        Args:
            name: ベース名
            timestamp: タイムスタンプ（オプション）

        Returns:
            生成されたパス
        """
        if timestamp is not None:
            filename = f"{name}_{timestamp:.6f}.{self.format}"
        else:
            filename = f"{name}.{self.format}"
        return self.output_dir / filename

    def get_field_config(self, field_name: str) -> Dict[str, Any]:
        """フィールドの可視化設定を取得

        Args:
            field_name: フィールド名

        Returns:
            設定辞書（存在しない場合はデフォルト設定）
        """
        return self.fields.get(
            field_name, {"enabled": True, "plot_types": ["scalar"], "alpha": 0.7}
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VisualizationConfig":
        """辞書から設定を作成

        Args:
            config: 設定辞書

        Returns:
            設定インスタンス
        """
        # 基本設定の取得
        base_config = {
            "output_dir": config.get("output_dir", "results/visualization"),
            "format": config.get("format", "png"),
            "dpi": config.get("dpi", 300),
            "colormap": config.get("colormap", "viridis"),
            "show_colorbar": config.get("show_colorbar", True),
            "show_axes": config.get("show_axes", True),
            "show_grid": config.get("show_grid", False),
        }

        # フィールド設定の取得とマージ
        fields = {}
        for field_name, field_config in config.get("fields", {}).items():
            fields[field_name] = {
                "enabled": field_config.get("enabled", True),
                "plot_types": field_config.get("plot_types", ["scalar"]),
                **field_config,
            }

        return cls(**base_config, fields=fields)


class DataSource:
    """データソースの基底クラス

    可視化対象のデータを提供するインターフェースを定義します。
    """

    @property
    def data(self) -> np.ndarray:
        """データ配列を取得"""
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, ...]:
        """データの形状を取得"""
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        """次元数を取得"""
        raise NotImplementedError


class Renderer:
    """レンダラーの基底クラス

    可視化の描画処理を担当する抽象基底クラスです。
    """

    def __init__(self, config: VisualizationConfig):
        """レンダラーを初期化

        Args:
            config: 可視化設定
        """
        self.config = config

    def render(
        self,
        data: Union[DataSource, np.ndarray],
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> Tuple[Any, Dict[str, Any]]:
        """データを描画

        Args:
            data: 描画するデータ
            view: 視点設定
            **kwargs: 追加の描画オプション

        Returns:
            (描画結果, メタデータの辞書)
        """
        raise NotImplementedError


class Exporter:
    """エクスポーターの基底クラス

    描画結果をファイルとして出力する抽象基底クラスです。
    """

    def __init__(self, config: VisualizationConfig):
        """エクスポーターを初期化

        Args:
            config: 可視化設定
        """
        self.config = config

    def export(
        self, figure: Any, filepath: Path, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """描画結果を出力

        Args:
            figure: 描画結果
            filepath: 出力パス
            metadata: メタデータ
        """
        raise NotImplementedError
