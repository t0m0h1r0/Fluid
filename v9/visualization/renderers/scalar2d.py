"""2Dスカラー場の可視化を提供するモジュール

このモジュールは、2次元スカラー場の可視化機能を実装します。
スカラー場のカラーマップ表示や等高線表示に対応します。
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet

from ..core.renderer import Renderer2D
from ..core.base import VisualizationConfig, ViewConfig


class Scalar2DRenderer(Renderer2D):
    """2Dスカラー場のレンダラー
    
    スカラー場をカラーマップや等高線として表示します。
    """
    
    def __init__(self, config: VisualizationConfig):
        """2Dスカラーレンダラーを初期化"""
        super().__init__(config)

    def render(
        self,
        data: np.ndarray,
        view: Optional[ViewConfig] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Dict[str, Any]]:
        """2Dスカラー場を描画

        Args:
            data: スカラー場データ
            view: 視点設定（3Dデータのスライス表示用）
            ax: 既存のAxes（Noneの場合は新規作成）
            **kwargs: 追加の描画オプション
                extent: データの表示範囲 [xmin, xmax, ymin, ymax]
                symmetric: 対称なカラースケール
                robust: ロバストな値域の使用
                cmap: カラーマップ名
                alpha: 透明度
                contour: 等高線の描画
                n_contours: 等高線の数
                contour_colors: 等高線の色
                contour_labels: 等高線ラベルの表示
                contour_label_fmt: ラベルのフォーマット
                levels: 明示的な等高線レベル

        Returns:
            (図, メタデータの辞書)のタプル
        """
        # 3Dデータの場合はスライスを抽出
        if data.ndim == 3 and view is not None:
            axis_map = {"xy": 2, "yz": 0, "xz": 1}
            primary_axis = view.slice_axes[0].lower()
            data = self.create_slice(data, axis_map[primary_axis], view.slice_positions[0])

        if data.ndim != 2:
            raise ValueError("2次元データが必要です")

        # 図とAxesの準備
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.figure

        # データの表示範囲を設定
        extent = kwargs.get("extent")
        if extent is None:
            ny, nx = data.shape
            extent = [0, nx, 0, ny]

        # カラーマップの設定
        cmap = self.create_colormap(
            kwargs.get("cmap"),
            kwargs.get("colors"),
            kwargs.get("reverse", False)
        )

        # データの正規化
        norm = self.create_normalizer(
            data,
            symmetric=kwargs.get("symmetric", False),
            robust=kwargs.get("robust", True)
        )

        # スカラー場の表示
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
            alpha=kwargs.get("alpha", 1.0),
            interpolation=kwargs.get("interpolation", "nearest")
        )

        # 等高線の描画
        contours = None
        if kwargs.get("contour", False):
            levels = kwargs.get("levels")
            if levels is None:
                n_contours = kwargs.get("n_contours", 10)
                levels = np.linspace(norm.vmin, norm.vmax, n_contours)

            contours = ax.contour(
                data.T,
                levels=levels,
                colors=kwargs.get("contour_colors", "k"),
                alpha=kwargs.get("contour_alpha", 0.7),
                extent=extent,
                linewidths=kwargs.get("contour_linewidth", 1.0)
            )

            # 等高線ラベルの追加
            if kwargs.get("contour_labels", True):
                label_fmt = kwargs.get("contour_label_fmt", "%.2f")
                ax.clabel(
                    contours,
                    inline=True,
                    fmt=label_fmt,
                    fontsize=kwargs.get("contour_label_size", 8)
                )

        # 軸の設定
        self.setup_2d_axes(
            ax,
            extent=extent,
            aspect=kwargs.get("aspect", "equal")
        )

        # カラーバーの追加
        if self.config.show_colorbar:
            label = kwargs.get("colorbar_label", "")
            self.setup_colorbar(
                im,
                ax,
                label=label,
                orientation=kwargs.get("colorbar_orientation", "vertical")
            )

        # 軸ラベルの追加
        if kwargs.get("xlabel"):
            ax.set_xlabel(kwargs["xlabel"])
        if kwargs.get("ylabel"):
            ax.set_ylabel(kwargs["ylabel"])

        # タイトルの追加
        if kwargs.get("title"):
            ax.set_title(kwargs["title"])

        # メタデータの収集
        metadata = self._collect_metadata(
            data=data,
            extent=extent,
            norm=norm,
            contours=contours,
            **kwargs
        )

        return fig, metadata

    def _collect_metadata(
        self,
        data: np.ndarray,
        extent: List[float],
        norm: plt.Normalize,
        contours: Optional[QuadContourSet],
        **kwargs
    ) -> Dict[str, Any]:
        """メタデータを収集

        Args:
            data: 表示データ
            extent: 表示範囲
            norm: 正規化オブジェクト
            contours: 等高線オブジェクト
            **kwargs: その他のパラメータ

        Returns:
            メタデータの辞書
        """
        metadata = {
            "data_range": {
                "min": float(norm.vmin),
                "max": float(norm.vmax),
            },
            "dimensions": data.shape,
            "extent": extent,
            "display_type": ["colormap"],
        }

        # 統計情報の追加
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0:
            metadata["statistics"] = {
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "median": float(np.median(valid_data))
            }

        # 等高線情報の追加
        if contours is not None:
            metadata["display_type"].append("contour")
            metadata["contour_levels"] = [float(level) for level in contours.levels]

        return metadata

    def create_slice_view(
        self,
        data: np.ndarray,
        view: ViewConfig,
        **kwargs
    ) -> Dict[str, Tuple[Figure, Dict[str, Any]]]:
        """複数のスライス表示を作成

        Args:
            data: 3Dデータ
            view: 視点設定
            **kwargs: 描画オプション

        Returns:
            スライス名をキーとする (図, メタデータ) のタプルの辞書
        """
        if data.ndim != 3:
            raise ValueError("3Dデータが必要です")

        result = {}
        axis_map = {"xy": 2, "yz": 0, "xz": 1}

        # 各スライスの表示を作成
        for axis_name, pos in zip(view.slice_axes, view.slice_positions):
            axis = axis_map[axis_name.lower()]
            slice_data = self.create_slice(data, axis, pos)

            # スライス固有の設定を適用
            slice_kwargs = kwargs.copy()
            if "title" in slice_kwargs:
                slice_kwargs["title"] = f"{slice_kwargs['title']} ({axis_name})"

            # スライスを描画
            fig, metadata = self.render(slice_data, view=None, **slice_kwargs)
            result[axis_name] = (fig, metadata)

        return result