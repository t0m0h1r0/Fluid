"""2Dベクトル場の可視化を提供するモジュール

このモジュールは、2次元ベクトル場の可視化機能を実装します。
ベクトル矢印表示、流線表示、マグニチュード表示などに対応します。
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.quiver import Quiver
from matplotlib.streamplot import StreamplotSet

from ..core.renderer import Renderer2D
from ..core.base import VisualizationConfig, ViewConfig


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
        view: Optional[ViewConfig] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Dict[str, Any]]:
        """2Dベクトル場を描画

        Args:
            vector_components: ベクトル場の各成分 [u, v]
            view: 視点設定（3Dデータのスライス表示用）
            ax: 既存のAxes（Noneの場合は新規作成）
            **kwargs: 追加の描画オプション
                extent: データの表示範囲 [xmin, xmax, ymin, ymax]
                density: ベクトル矢印の密度
                scale: ベクトル矢印のスケール
                width: 矢印の幅
                headwidth: 矢印の頭の幅
                headlength: 矢印の頭の長さ
                headaxislength: 矢印の頭の軸長
                minshaft: 最小シャフト長
                minlength: 最小矢印長
                color: ベクトルの色
                magnitude_colors: 大きさに応じた色付け
                alpha: 透明度
                streamlines: 流線の描画
                n_streamlines: 流線の数
                streamline_density: 流線の密度
                streamline_color: 流線の色
                streamline_width: 流線の幅
                streamline_alpha: 流線の透明度

        Returns:
            (図, メタデータの辞書)のタプル
        """
        if len(vector_components) != 2:
            raise ValueError("2次元ベクトル場には2つの成分が必要です")

        # 3Dデータの場合はスライスを抽出
        if vector_components[0].ndim == 3 and view is not None:
            axis_map = {"xy": 2, "yz": 0, "xz": 1}
            primary_axis = view.slice_axes[0].lower()
            axis = axis_map[primary_axis]
            vector_components = [
                self.create_slice(comp, axis, view.slice_positions[0])
                for comp in vector_components
            ]

        u, v = vector_components
        if u.shape != v.shape or u.ndim != 2:
            raise ValueError("無効なベクトル場の形状です")

        # 図とAxesの準備
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.figure

        # グリッドの生成
        ny, nx = u.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # ベクトルの大きさを計算
        magnitude = np.sqrt(u**2 + v**2)
        norm = self.create_normalizer(
            magnitude,
            robust=kwargs.get("robust", True)
        )

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
                headwidth=kwargs.get("headwidth", 3),
                headlength=kwargs.get("headlength", 5),
                headaxislength=kwargs.get("headaxislength", 4.5),
                minshaft=kwargs.get("minshaft", 1),
                minlength=kwargs.get("minlength", 0.1),
                alpha=kwargs.get("alpha", 1.0),
            )
        else:
            # 単色表示
            color = kwargs.get("color", "k")
            q = ax.quiver(
                X[::skip, ::skip],
                Y[::skip, ::skip],
                u[::skip, ::skip],
                v[::skip, ::skip],
                color=color,
                scale=scale,
                width=kwargs.get("width", 0.005),
                headwidth=kwargs.get("headwidth", 3),
                headlength=kwargs.get("headlength", 5),
                headaxislength=kwargs.get("headaxislength", 4.5),
                minshaft=kwargs.get("minshaft", 1),
                minlength=kwargs.get("minlength", 0.1),
                alpha=kwargs.get("alpha", 1.0),
            )

        # 流線の描画
        streamlines = None
        if kwargs.get("streamlines", False):
            streamline_density = kwargs.get("streamline_density", 2)
            streamline_color = kwargs.get("streamline_color")
            
            if streamline_color is None:
                # 大きさに応じた色付け
                streamline_color = magnitude
                
            streamlines = ax.streamplot(
                x,
                y,
                u,
                v,
                density=streamline_density,
                color=streamline_color,
                cmap=kwargs.get("streamline_cmap", plt.cm.viridis),
                linewidth=kwargs.get("streamline_width", 1),
                arrowsize=kwargs.get("streamline_arrow_size", 1),
                alpha=kwargs.get("streamline_alpha", 0.7),
            )

        # 軸の設定
        extent = kwargs.get("extent")
        if extent is None:
            extent = [0, nx, 0, ny]
        self.setup_2d_axes(ax, extent)

        # カラーバーの追加
        if kwargs.get("magnitude_colors", True) and self.config.show_colorbar:
            label = kwargs.get("colorbar_label", "Velocity magnitude")
            self.setup_colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis),
                ax,
                label=label
            )

        # スケールバーの追加
        if kwargs.get("show_scale", True):
            scale_label = kwargs.get("scale_label", "1 unit")
            ax.quiverkey(
                q,
                0.9,
                0.9,
                1.0,
                scale_label,
                labelpos="E",
                coordinates="figure"
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
            u=u,
            v=v,
            magnitude=magnitude,
            quiver=q,
            streamlines=streamlines,
            **kwargs
        )

        return fig, metadata

    def _collect_metadata(
        self,
        u: np.ndarray,
        v: np.ndarray,
        magnitude: np.ndarray,
        quiver: Quiver,
        streamlines: Optional[StreamplotSet],
        **kwargs
    ) -> Dict[str, Any]:
        """メタデータを収集

        Args:
            u: x方向の速度成分
            v: y方向の速度成分
            magnitude: ベクトルの大きさ
            quiver: ベクトル矢印オブジェクト
            streamlines: 流線オブジェクト
            **kwargs: その他のパラメータ

        Returns:
            メタデータの辞書
        """
        metadata = {
            "data_range": {
                "min_magnitude": float(np.min(magnitude)),
                "max_magnitude": float(np.max(magnitude)),
                "mean_magnitude": float(np.mean(magnitude)),
                "u_range": [float(np.min(u)), float(np.max(u))],
                "v_range": [float(np.min(v)), float(np.max(v))],
            },
            "dimensions": u.shape,
            "display_type": ["quiver"],
        }

        # ベクトル場の特性を追加
        valid_mask = np.isfinite(u) & np.isfinite(v)
        if np.any(valid_mask):
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            metadata["statistics"] = {
                "divergence": float(np.mean(np.gradient(u_valid, axis=0) + np.gradient(v_valid, axis=1))),
                "curl": float(np.mean(np.gradient(v_valid, axis=0) - np.gradient(u_valid, axis=1))),
                "kinetic_energy": float(0.5 * np.mean(u_valid**2 + v_valid**2)),
                "enstrophy": float(0.5 * np.mean((np.gradient(v_valid, axis=0) - np.gradient(u_valid, axis=1))**2))
            }

        # 流線情報の追加
        if streamlines is not None:
            metadata["display_type"].append("streamlines")
            metadata["streamline_density"] = kwargs.get("streamline_density", 2)

        return metadata

    def create_slice_view(
        self,
        vector_components: List[np.ndarray],
        view: ViewConfig,
        **kwargs
    ) -> Dict[str, Tuple[Figure, Dict[str, Any]]]:
        """複数のスライス表示を作成

        Args:
            vector_components: 3Dベクトル場の各成分 [u, v, w]
            view: 視点設定
            **kwargs: 描画オプション

        Returns:
            スライス名をキーとする (図, メタデータ) のタプルの辞書
        """
        if any(comp.ndim != 3 for comp in vector_components):
            raise ValueError("3Dデータが必要です")

        result = {}
        axis_map = {"xy": 2, "yz": 0, "xz": 1}
        component_map = {
            "xy": ([0, 1], ["u", "v"]),
            "yz": ([1, 2], ["v", "w"]),
            "xz": ([0, 2], ["u", "w"])
        }

        # 各スライスの表示を作成
        for axis_name, pos in zip(view.slice_axes, view.slice_positions):
            axis = axis_map[axis_name.lower()]
            comp_indices, comp_names = component_map[axis_name.lower()]

            # 該当する成分のスライスを取得
            slice_components = [
                self.create_slice(vector_components[i], axis, pos)
                for i in comp_indices
            ]

            # スライス固有の設定を適用
            slice_kwargs = kwargs.copy()
            if "title" in slice_kwargs:
                slice_kwargs["title"] = f"{slice_kwargs['title']} ({axis_name})"
            
            # 軸ラベルを更新
            slice_kwargs["xlabel"] = comp_names[0]
            slice_kwargs["ylabel"] = comp_names[1]

            # スライスを描画
            fig, metadata = self.render(slice_components, view=None, **slice_kwargs)
            result[axis_name] = (fig, metadata)

        return result

    def compute_derivatives(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """ベクトル場の微分量を計算

        Args:
            u: x方向の速度成分
            v: y方向の速度成分

        Returns:
            微分量の辞書:
            - divergence: 発散
            - vorticity: 渦度
            - shear: せん断歪み
            - strain: 伸縮歪み
        """
        # 勾配の計算
        dudx = np.gradient(u, axis=0)
        dudy = np.gradient(u, axis=1)
        dvdx = np.gradient(v, axis=0)
        dvdy = np.gradient(v, axis=1)

        return {
            "divergence": dudx + dvdy,  # ∇・v
            "vorticity": dvdx - dudy,   # ∇×v
            "shear": dudy + dvdx,       # せん断歪み
            "strain": dudx - dvdy       # 伸縮歪み
        }

    def compute_derived_quantities(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> Dict[str, float]:
        """ベクトル場の導出量を計算

        Args:
            u: x方向の速度成分
            v: y方向の速度成分

        Returns:
            導出量の辞書:
            - kinetic_energy: 運動エネルギー
            - enstrophy: エンストロフィー
            - palinstrophy: パリンストロフィー
        """
        # 渦度の計算
        vorticity = np.gradient(v, axis=0) - np.gradient(u, axis=1)
        
        # 渦度勾配の計算
        vort_grad_x = np.gradient(vorticity, axis=0)
        vort_grad_y = np.gradient(vorticity, axis=1)

        return {
            "kinetic_energy": float(0.5 * np.mean(u**2 + v**2)),
            "enstrophy": float(0.5 * np.mean(vorticity**2)),
            "palinstrophy": float(0.5 * np.mean(vort_grad_x**2 + vort_grad_y**2))
        }