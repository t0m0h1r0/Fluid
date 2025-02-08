"""2Dベクトル場の可視化を提供するモジュール

このモジュールは、2次元ベクトル場の可視化機能を実装します。
ベクトル矢印や流線の表示に対応します。
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt

from ..core.renderer import Renderer2D
from ..core.base import VisualizationConfig


class Vector2DRenderer(Renderer2D):
    """2Dベクトル場のレンダラー

    ベクトル場を矢印や流線として表示します。
    """

    def __init__(self, config: VisualizationConfig):
        """2Dベクトルレンダラーを初期化"""
        super().__init__(config)

    def render(
        self,
        vector_components: List[np.ndarray],
        extent: Optional[Tuple[float, float, float, float]] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """2Dベクトル場を描画

        Args:
            vector_components: ベクトル場の各成分 [u, v]
            extent: データの表示範囲 [xmin, xmax, ymin, ymax]
            **kwargs: 追加の描画オプション
                - density: ベクトル矢印の密度
                - scale: ベクトル矢印のスケール
                - width: 矢印の幅
                - streamlines: 流線の描画
                - n_streamlines: 流線の数
                - color: ベクトルの色
                - magnitude_colors: 大きさに応じた色付け
                - alpha: 透明度

        Returns:
            (Figure, メタデータの辞書)
        """
        if len(vector_components) != 2:
            raise ValueError("2次元ベクトル場には2つの成分が必要です")

        u, v = vector_components
        if u.shape != v.shape or u.ndim != 2:
            raise ValueError("無効なベクトル場の形状です")

        # 図の作成
        fig, ax = self.create_figure()

        # グリッドの生成
        ny, nx = u.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        # ベクトルの大きさを計算
        magnitude = np.sqrt(u**2 + v**2)
        norm = self.create_normalizer(magnitude)

        # 表示密度の設定
        density = kwargs.get("density", 20)
        scale = kwargs.get("scale", 1.0)
        skip = max(1, min(nx, ny) // density)

        # ベクトル場の表示
        if kwargs.get("magnitude_colors", True):
            # 大きさに応じた色付け
            colors = plt.cm.viridis(norm(magnitude[::skip, ::skip]))
            q = ax.quiver(
                X[::skip, ::skip],
                Y[::skip, ::skip],
                u[::skip, ::skip],
                v[::skip, ::skip],
                color=colors,
                scale=scale,
                width=kwargs.get("width", 0.005),
                alpha=kwargs.get("alpha", 1.0),
            )
        else:
            # 単色表示
            q = ax.quiver(
                X[::skip, ::skip],
                Y[::skip, ::skip],
                u[::skip, ::skip],
                v[::skip, ::skip],
                color=kwargs.get("color", "k"),
                scale=scale,
                width=kwargs.get("width", 0.005),
                alpha=kwargs.get("alpha", 1.0),
            )

        # 流線の描画
        if kwargs.get("streamlines", False):
            n_streamlines = kwargs.get("n_streamlines", 50)

            # 開始点をランダムに選択
            start_points = np.random.rand(n_streamlines, 2) * np.array([nx, ny])

            # 流線を描画
            streamlines = ax.streamplot(
                X,
                Y,
                u,
                v,
                color=magnitude,
                cmap=kwargs.get("cmap", self.config.colormap),
                density=kwargs.get("streamline_density", 1),
                linewidth=kwargs.get("streamline_width", 1),
                arrowsize=kwargs.get("streamline_arrow_size", 1),
                alpha=kwargs.get("streamline_alpha", 0.7),
            )

        # 軸の設定
        self.setup_2d_axes(ax, extent)

        # カラーバーの追加（大きさに応じた色付けの場合）
        if kwargs.get("magnitude_colors", True) and self.config.show_colorbar:
            cbar = self.setup_colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=self.config.colormap),
                ax,
                "Velocity magnitude",
            )

        # スケールバーの追加
        if kwargs.get("show_scale", True):
            ax.quiverkey(q, 0.9, 0.9, 1.0, "1 unit", labelpos="E", coordinates="figure")

        # メタデータの収集
        metadata = {
            "data_range": {
                "min_magnitude": float(np.min(magnitude)),
                "max_magnitude": float(np.max(magnitude)),
            },
            "dimensions": u.shape,
            "display_type": ["quiver"],
        }

        if kwargs.get("streamlines", False):
            metadata["display_type"].append("streamlines")

        if kwargs.get("magnitude_colors", True):
            metadata["display_type"].append("magnitude_colors")

        return fig, metadata

    def _compute_derived_quantities(
        self, u: np.ndarray, v: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """派生量を計算

        Args:
            u: x方向の速度成分
            v: y方向の速度成分

        Returns:
            派生量の辞書
        """
        # 渦度の計算
        dudy = np.gradient(u, axis=0)
        dvdx = np.gradient(v, axis=1)
        vorticity = dvdx - dudy

        # 発散の計算
        dudx = np.gradient(u, axis=1)
        dvdy = np.gradient(v, axis=0)
        divergence = dudx + dvdy

        return {"vorticity": vorticity, "divergence": divergence}
