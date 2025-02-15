"""レンダリングエンジンの基底クラスを提供するモジュール

このモジュールは、2D/3Dレンダリングの基本機能を提供します。
matplotlibを使用した描画処理の共通部分を実装します。
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List, Union, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import colormaps
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.collections import QuadMesh

if TYPE_CHECKING:
    from .base import VisualizationConfig, ViewConfig


class BaseRenderer(ABC):
    """レンダラーの基底クラス

    全てのレンダラーに共通の機能を提供します。
    """

    def __init__(self, config: "VisualizationConfig"):
        """レンダラーを初期化

        Args:
            config: 可視化設定
        """
        self.config = config

    def create_figure(
        self, projection: Optional[str] = None, figsize: Tuple[float, float] = (10, 8)
    ) -> Tuple[Figure, Axes]:
        """図とAxesを作成

        Args:
            projection: プロジェクションの種類
            figsize: 図のサイズ

        Returns:
            (図, Axes)のタプル
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=projection)

        # 軸の設定
        if not self.config.show_axes:
            ax.set_axis_off()

        # グリッドの設定
        if self.config.show_grid:
            ax.grid(True)

        return fig, ax

    def setup_colorbar(
        self,
        mappable: Union[QuadMesh, plt.cm.ScalarMappable],
        ax: Axes,
        label: str = "",
        orientation: str = "vertical",
    ) -> Optional[Any]:
        """カラーバーを設定

        Args:
            mappable: カラーマップを持つオブジェクト
            ax: 対象のAxes
            label: カラーバーのラベル
            orientation: カラーバーの向き

        Returns:
            作成されたカラーバー
        """
        if self.config.show_colorbar:
            return plt.colorbar(mappable, ax=ax, label=label, orientation=orientation)
        return None

    def compute_data_range(
        self,
        data: np.ndarray,
        symmetric: bool = False,
        robust: bool = True,
        percentile: float = 2.0,
    ) -> Tuple[float, float]:
        """データの範囲を計算

        Args:
            data: 入力データ
            symmetric: 対称な範囲にするかどうか
            robust: 外れ値に対してロバストな範囲を使用するか
            percentile: ロバスト範囲計算時のパーセンタイル

        Returns:
            (最小値, 最大値)のタプル
        """
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return 0.0, 1.0

        if robust:
            vmin = np.percentile(valid_data, percentile)
            vmax = np.percentile(valid_data, 100 - percentile)
        else:
            vmin = np.min(valid_data)
            vmax = np.max(valid_data)

        if symmetric:
            abs_max = max(abs(vmin), abs(vmax))
            return -abs_max, abs_max

        return vmin, vmax

    def create_normalizer(
        self, data: np.ndarray, symmetric: bool = False, robust: bool = True
    ) -> Normalize:
        """データの正規化オブジェクトを作成

        Args:
            data: 入力データ
            symmetric: 対称な範囲にするかどうか
            robust: 外れ値に対してロバストな範囲を使用するか

        Returns:
            正規化オブジェクト
        """
        vmin, vmax = self.compute_data_range(data, symmetric, robust)
        return Normalize(vmin=vmin, vmax=vmax)

    def create_colormap(
        self,
        name: Optional[str] = None,
        colors: Optional[List[str]] = None,
        reverse: bool = False,
    ) -> LinearSegmentedColormap:
        """カラーマップを作成

        Args:
            name: カラーマップ名（Noneの場合はcolorsを使用）
            colors: カラーのリスト
            reverse: カラーマップを反転するかどうか

        Returns:
            作成されたカラーマップ
        """
        if colors is not None:
            cmap = LinearSegmentedColormap.from_list("custom", colors)
        else:
            cmap = colormaps[name or self.config.colormap]

        if reverse:
            return LinearSegmentedColormap.from_list(
                f"{cmap.name}_r", cmap(np.linspace(1, 0, cmap.N))
            )
        return cmap

    @abstractmethod
    def render(self, *args, **kwargs) -> Tuple[Figure, Dict[str, Any]]:
        """描画を実行

        Args:
            *args: 位置引数
            **kwargs: キーワード引数

        Returns:
            (図, メタデータの辞書)のタプル
        """
        pass


class Renderer2D(BaseRenderer):
    """2Dレンダラーの基底クラス"""

    def setup_2d_axes(
        self,
        ax: Axes,
        extent: Optional[Tuple[float, float, float, float]] = None,
        aspect: str = "equal",
    ) -> None:
        """2D軸を設定

        Args:
            ax: 対象のAxes
            extent: データの表示範囲 [xmin, xmax, ymin, ymax]
            aspect: アスペクト比の設定
        """
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        ax.set_aspect(aspect)

        if self.config.show_axes:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

    def create_slice(self, data: np.ndarray, axis: int, position: float) -> np.ndarray:
        """3Dデータから2Dスライスを抽出

        Args:
            data: 3Dデータ
            axis: スライスする軸
            position: スライス位置（0-1）

        Returns:
            抽出された2Dスライス
        """
        if data.ndim != 3:
            raise ValueError("3Dデータが必要です")

        # スライスインデックスを計算
        idx = int(position * (data.shape[axis] - 1))

        # データのスライスを取得
        slices = [slice(None)] * 3
        slices[axis] = slice(idx, idx + 1)
        slice_data = data[tuple(slices)]

        # 余分な次元を削除
        return np.squeeze(slice_data)


class Renderer3D(BaseRenderer):
    """3Dレンダラーの基底クラス"""

    def setup_3d_axes(
        self,
        ax: Axes,
        view: Optional["ViewConfig"] = None,
        bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """3D軸を設定

        Args:
            ax: 対象のAxes
            view: 視点設定
            bounds: データの境界 [(xmin, ymin, zmin), (xmax, ymax, zmax)]
        """
        if view is not None:
            ax.view_init(elev=view.elevation, azim=view.azimuth)
            ax.dist = view.distance

        if bounds is not None:
            min_point, max_point = bounds
            ax.set_xlim(min_point[0], max_point[0])
            ax.set_ylim(min_point[1], max_point[1])
            ax.set_zlim(min_point[2], max_point[2])

        if self.config.show_axes:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
