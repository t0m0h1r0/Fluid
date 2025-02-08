"""2Dスカラー場の可視化を提供するモジュール

このモジュールは、2次元スカラー場の可視化機能を実装します。
等高線図やカラーマップ表示に対応します。
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from ..core.renderer import Renderer2D
from ..core.base import VisualizationConfig


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
        extent: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """2Dスカラー場を描画

        Args:
            data: スカラー場データ
            extent: データの表示範囲 [xmin, xmax, ymin, ymax]
            **kwargs: 追加の描画オプション
                - symmetric: 対称なカラースケール
                - contour: 等高線の描画
                - n_contours: 等高線の数
                - contour_colors: 等高線の色
                - cmap: カラーマップ名
                - alpha: 透明度

        Returns:
            (Figure, メタデータの辞書)
        """
        if data.ndim != 2:
            raise ValueError("2次元データが必要です")

        # 図の作成
        fig, ax = self.create_figure()

        # データの範囲を計算
        symmetric = kwargs.get("symmetric", False)
        norm = self.create_normalizer(data, symmetric)

        # カラーマップの設定
        cmap = kwargs.get("cmap", self.config.colormap)
        alpha = kwargs.get("alpha", 1.0)

        # データの表示
        im = ax.imshow(
            data.T, origin="lower", extent=extent, norm=norm, cmap=cmap, alpha=alpha
        )

        # 等高線の描画
        if kwargs.get("contour", False):
            n_contours = kwargs.get("n_contours", 10)
            contour_colors = kwargs.get("contour_colors", "k")

            # 等高線レベルの計算
            levels = np.linspace(norm.vmin, norm.vmax, n_contours)

            # 等高線の描画
            cs = ax.contour(
                data.T, levels=levels, colors=contour_colors, alpha=0.7, extent=extent
            )

            # 等高線ラベルの追加
            if kwargs.get("contour_labels", True):
                ax.clabel(cs, inline=True, fontsize=8)

        # 軸の設定
        self.setup_2d_axes(ax, extent)

        # カラーバーの追加
        if self.config.show_colorbar:
            self.setup_colorbar(im, ax)

        # メタデータの収集
        metadata = {
            "data_range": {"min": float(norm.vmin), "max": float(norm.vmax)},
            "dimensions": data.shape,
            "display_type": ["colormap"],
        }

        if kwargs.get("contour", False):
            metadata["display_type"].append("contour")
            metadata["contour_levels"] = levels.tolist()

        return fig, metadata
