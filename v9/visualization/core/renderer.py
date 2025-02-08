"""レンダリングエンジンの基底クラスを提供するモジュール

このモジュールは、様々な種類のレンダラーに共通の機能を提供します。
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .base import VisualizationConfig, ViewConfig


class BaseRenderer(ABC):
    """レンダラーの基底クラス

    全てのレンダラーに共通の機能を提供します。
    """

    def __init__(self, config: VisualizationConfig):
        """レンダラーを初期化

        Args:
            config: 可視化設定
        """
        self.config = config

    def create_figure(
        self, projection: Optional[str] = None, figsize: Tuple[float, float] = (10, 8)
    ) -> Tuple[plt.Figure, plt.Axes]:
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
        self, mappable: plt.cm.ScalarMappable, ax: plt.Axes, label: str = ""
    ) -> None:
        """カラーバーを設定

        Args:
            mappable: カラーマップを持つオブジェクト
            ax: 対象のAxes
            label: カラーバーのラベル
        """
        if self.config.show_colorbar:
            plt.colorbar(mappable, ax=ax, label=label)

    def compute_data_range(
        self, data: np.ndarray, symmetric: bool = False
    ) -> Tuple[float, float]:
        """データの範囲を計算

        Args:
            data: 入力データ
            symmetric: 対称な範囲にするかどうか

        Returns:
            (最小値, 最大値)のタプル
        """
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return 0.0, 1.0

        vmin = np.min(valid_data)
        vmax = np.max(valid_data)

        if symmetric:
            abs_max = max(abs(vmin), abs(vmax))
            return -abs_max, abs_max

        return vmin, vmax

    def create_normalizer(self, data: np.ndarray, symmetric: bool = False) -> Normalize:
        """データの正規化オブジェクトを作成

        Args:
            data: 入力データ
            symmetric: 対称な範囲にするかどうか

        Returns:
            正規化オブジェクト
        """
        vmin, vmax = self.compute_data_range(data, symmetric)
        return Normalize(vmin=vmin, vmax=vmax)

    @abstractmethod
    def render(self, *args, **kwargs) -> Tuple[plt.Figure, Dict[str, Any]]:
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
        self, ax: plt.Axes, extent: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        """2D軸を設定

        Args:
            ax: 対象のAxes
            extent: データの表示範囲 [xmin, xmax, ymin, ymax]
        """
        if extent is not None:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

        ax.set_aspect("equal")

        if self.config.show_axes:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")


class Renderer3D(BaseRenderer):
    """3Dレンダラーの基底クラス"""

    def setup_3d_axes(
        self,
        ax: plt.Axes,
        view: Optional[ViewConfig] = None,
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
