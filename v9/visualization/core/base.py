from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Dict, Any, Optional, Tuple, List, Union
import numpy as np
import numpy.typing as npt


@dataclass
class ViewConfig:
    """可視化の表示設定

    Attributes:
        elevation: 仰角（度）
        azimuth: 方位角（度）
        distance: 視点距離
        focal_point: 注視点座標
        slice_position: スライス位置（0から1の間の値）
    """

    elevation: float = 30.0
    azimuth: float = 45.0
    distance: float = 10.0
    focal_point: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    slice_position: Union[float, npt.NDArray[np.float64]] = 0.5

    def __getitem__(self, key):
        """スライス用の互換性のためのメソッド"""
        if key == "slice_position":
            return self.slice_position
        raise KeyError(f"Invalid key: {key}")


@dataclass
class VectorPlotConfig:
    """ベクトル場の表示設定"""

    density: int = 20
    scale: float = 1.0
    width: float = 0.005
    show_magnitude: bool = True
    streamlines: bool = False
    n_streamlines: int = 50


@dataclass
class ScalarPlotConfig:
    """スカラー場の表示設定"""

    interpolation: str = "nearest"
    contour: bool = False
    n_contours: int = 10
    symmetric: bool = False


@dataclass
class VolumePlotConfig:
    """3D表示の設定"""

    opacity: float = 0.7
    slice_positions: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    show_isosurfaces: bool = True
    n_isosurfaces: int = 5


@dataclass
class InterfacePlotConfig:
    """Level Set場の表示設定"""

    color: str = "black"
    width: float = 2.0
    filled: bool = True
    phase_colors: List[str] = field(default_factory=lambda: ["lightblue", "white"])


@dataclass
class VisualizationConfig:
    """可視化の基本設定

    Attributes:
        output_dir: 出力ディレクトリ
        format: 出力フォーマット
        dpi: 解像度
        colormap: デフォルトのカラーマップ
        show_colorbar: カラーバーの表示
        show_axes: 軸の表示
        show_grid: グリッドの表示
    """

    output_dir: Union[str, Path] = "results/visualization"
    format: str = "png"
    dpi: int = 300
    colormap: str = "viridis"
    show_colorbar: bool = True
    show_axes: bool = True
    show_grid: bool = False

    # 各種プロット設定
    vector_plot: VectorPlotConfig = field(default_factory=VectorPlotConfig)
    scalar_plot: ScalarPlotConfig = field(default_factory=ScalarPlotConfig)
    volume_plot: VolumePlotConfig = field(default_factory=VolumePlotConfig)
    interface_plot: InterfacePlotConfig = field(default_factory=InterfacePlotConfig)

    # フィールドごとの可視化設定
    fields: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "velocity": {"enabled": True, "plot_types": ["vector", "magnitude"]},
            "pressure": {"enabled": True, "plot_types": ["scalar", "contour"]},
            "levelset": {"enabled": True, "plot_types": ["interface", "contour"]},
        }
    )

    def __post_init__(self):
        """設定の後処理"""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # デフォルト値の追加
        for field_name, field_config in self.fields.items():
            if "enabled" not in field_config:
                field_config["enabled"] = True
            if "plot_types" not in field_config:
                field_config["plot_types"] = []

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

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VisualizationConfig":
        """辞書から設定を作成

        Args:
            config: 設定辞書

        Returns:
            設定インスタンス
        """
        # 各プロット設定のデフォルト値を保持
        vector_config = VectorPlotConfig(**config.get("vector_plot", {}))
        scalar_config = ScalarPlotConfig(**config.get("scalar_plot", {}))
        volume_config = VolumePlotConfig(**config.get("volume_plot", {}))
        interface_config = InterfacePlotConfig(**config.get("interface_plot", {}))

        return cls(
            output_dir=config.get("output_dir", "results/visualization"),
            format=config.get("format", "png"),
            dpi=config.get("dpi", 300),
            colormap=config.get("colormap", "viridis"),
            show_colorbar=config.get("show_colorbar", True),
            show_axes=config.get("show_axes", True),
            show_grid=config.get("show_grid", False),
            vector_plot=vector_config,
            scalar_plot=scalar_config,
            volume_plot=volume_config,
            interface_plot=interface_config,
            fields=config.get("fields", {}),
        )


class DataSource(Protocol):
    """データソースのプロトコル

    可視化対象のデータを提供するインターフェースを定義します。
    """

    @property
    def data(self) -> np.ndarray:
        """データ配列を取得"""
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """データの形状を取得"""
        ...

    @property
    def ndim(self) -> int:
        """次元数を取得"""
        ...


class Renderer(ABC):
    """レンダラーの基底クラス

    可視化の実際の描画処理を担当する抽象基底クラスです。
    """

    def __init__(self, config: VisualizationConfig):
        """レンダラーを初期化"""
        self.config = config

    @abstractmethod
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
        pass


class Exporter(ABC):
    """エクスポーターの基底クラス

    描画結果をファイルとして出力する抽象基底クラスです。
    """

    def __init__(self, config: VisualizationConfig):
        """エクスポーターを初期化"""
        self.config = config

    @abstractmethod
    def export(
        self, figure: Any, filepath: Path, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """描画結果を出力

        Args:
            figure: 描画結果
            filepath: 出力パス
            metadata: メタデータ
        """
        pass
