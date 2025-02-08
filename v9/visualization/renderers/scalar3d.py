"""3Dスカラー場の可視化を提供するモジュール

このモジュールは、3次元スカラー場の可視化機能を実装します。
等値面表示、断面表示、ボリュームレンダリングなどに対応します。
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from skimage import measure

    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False

from ..core.renderer import Renderer3D
from ..core.base import VisualizationConfig, ViewConfig


class Scalar3DRenderer(Renderer3D):
    """3Dスカラー場のレンダラー

    スカラー場を等値面や断面として表示します。
    """

    def __init__(self, config: VisualizationConfig):
        """3Dスカラーレンダラーを初期化"""
        super().__init__(config)

    def render(
        self,
        data: np.ndarray,
        view: Optional[ViewConfig] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Tuple[Figure, Dict[str, Any]]:
        """3Dスカラー場を描画

        Args:
            data: スカラー場データ
            view: 視点設定
            ax: 既存のAxes（Noneの場合は新規作成）
            **kwargs: 追加の描画オプション
                isovalues: 等値面の値のリスト
                isovalue_colors: 等値面の色のリスト
                opacity: 不透明度
                slice_positions: 各軸でのスライス位置 [x, y, z]
                slice_axes: 表示する断面の軸 ["xy", "yz", "xz"]
                cmap: カラーマップ名
                symmetric: 対称なカラースケール
                robust: ロバストな値域の使用
                volume_render: ボリュームレンダリングの使用
                volume_alpha: ボリュームレンダリングの透明度関数

        Returns:
            (図, メタデータの辞書)のタプル
        """
        if data.ndim != 3:
            raise ValueError("3次元データが必要です")

        # 図とAxesの準備
        if ax is None:
            fig, ax = self.create_figure(projection="3d")
        else:
            fig = ax.figure
            if not isinstance(ax, Axes3D):
                raise ValueError("3D Axesが必要です")

        # カラーマップの設定
        cmap = self.create_colormap(
            kwargs.get("cmap"), kwargs.get("colors"), kwargs.get("reverse", False)
        )

        # データの正規化
        norm = self.create_normalizer(
            data,
            symmetric=kwargs.get("symmetric", False),
            robust=kwargs.get("robust", True),
        )

        # メタデータの初期化
        metadata = {
            "data_range": {"min": float(norm.vmin), "max": float(norm.vmax)},
            "dimensions": data.shape,
            "display_type": [],
        }

        # 等値面の描画
        if "isovalues" in kwargs:
            self._render_isosurfaces(
                ax=ax,
                data=data,
                isovalues=kwargs["isovalues"],
                colors=kwargs.get("isovalue_colors"),
                opacity=kwargs.get("opacity", 0.3),
                cmap=cmap,
                norm=norm,
                metadata=metadata,
            )

        # 断面の描画
        if view is not None:
            self._render_slices(
                ax=ax,
                data=data,
                view=view,
                cmap=cmap,
                norm=norm,
                opacity=kwargs.get("opacity", 0.7),
                metadata=metadata,
                **kwargs,
            )

        # ボリュームレンダリング
        if kwargs.get("volume_render", False):
            self._render_volume(
                ax=ax,
                data=data,
                cmap=cmap,
                norm=norm,
                alpha=kwargs.get("volume_alpha"),
                metadata=metadata,
            )

        # 視点の設定
        if view is not None:
            ax.view_init(elev=view.elevation, azim=view.azimuth)
            ax.dist = view.distance

        # 軸の設定
        if self.config.show_axes:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
        else:
            ax.set_axis_off()

        # グリッドの設定
        if self.config.show_grid:
            ax.grid(True)

        # カラーバーの追加
        if self.config.show_colorbar:
            label = kwargs.get("colorbar_label", "")
            self.setup_colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax, label=label
            )

        # タイトルの追加
        if kwargs.get("title"):
            ax.set_title(kwargs["title"])

        return fig, metadata

    def _render_isosurfaces(
        self,
        ax: Axes3D,
        data: np.ndarray,
        isovalues: List[float],
        colors: Optional[List[str]],
        opacity: float,
        cmap: plt.cm.ScalarMappable,
        norm: plt.Normalize,
        metadata: Dict[str, Any],
    ) -> None:
        """等値面を描画

        Args:
            ax: 3D Axes
            data: スカラー場データ
            isovalues: 等値面の値のリスト
            colors: 等値面の色のリスト
            opacity: 不透明度
            cmap: カラーマップ
            norm: 正規化オブジェクト
            metadata: 更新するメタデータ
        """
        if not HAVE_SKIMAGE:
            return

        metadata["display_type"].append("isosurface")
        metadata["isovalues"] = []

        for i, isovalue in enumerate(isovalues):
            try:
                # 等値面の計算
                verts, faces, _, _ = measure.marching_cubes(data, isovalue)

                # 等値面をポリゴンとして描画
                mesh = Poly3DCollection(verts[faces])

                # 色の設定
                if colors is not None and i < len(colors):
                    # カラーが文字列の場合、RGBAに変換
                    color = colors[i]
                    if isinstance(color, str):
                        color = plt.cm.colors.to_rgba(color)
                    elif isinstance(color, (list, np.ndarray)):
                        # すでにRGBA形式であることを確認
                        if len(color) not in [3, 4]:
                            raise ValueError(f"Invalid color format: {color}")
                    else:
                        raise ValueError(f"Unsupported color type: {type(color)}")
                else:
                    color = cmap(norm(isovalue))

                mesh.set_facecolor(color)
                mesh.set_alpha(opacity)

                # 等値面の追加
                ax.add_collection3d(mesh)

                # メタデータの更新
                metadata["isovalues"].append(
                    {
                        "value": float(isovalue),
                        "vertices": len(verts),
                        "faces": len(faces),
                    }
                )

            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.warning(f"等値面の生成エラー ({isovalue}): {e}")

    def _render_slices(
        self,
        ax: Axes3D,
        data: np.ndarray,
        view: ViewConfig,
        cmap: plt.cm.ScalarMappable,
        norm: plt.Normalize,
        opacity: float,
        metadata: Dict[str, Any],
        **kwargs,
    ) -> None:
        """断面を描画

        Args:
            ax: 3D Axes
            data: スカラー場データ
            view: 視点設定
            cmap: カラーマップ
            norm: 正規化オブジェクト
            opacity: 不透明度
            metadata: 更新するメタデータ
            **kwargs: 追加の描画オプション
        """
        if "slices" not in metadata["display_type"]:
            metadata["display_type"].append("slices")
            metadata["slices"] = []

        # 座標グリッドの生成
        nx, ny, nz = data.shape
        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]

        # 各断面の描画
        axis_map = {"xy": 2, "yz": 0, "xz": 1}
        for axis_name, pos in zip(view.slice_axes, view.slice_positions):
            try:
                axis = axis_map[axis_name.lower()]
                idx = int(pos * (data.shape[axis] - 1))

                # インデックスでのスライスを取得
                if axis == 0:
                    X, Y, Z = x[idx, :, :], y[idx, :, :], z[idx, :, :]
                    slice_data = data[idx, :, :]
                elif axis == 1:
                    X, Y, Z = x[:, idx, :], y[:, idx, :], z[:, idx, :]
                    slice_data = data[:, idx, :]
                else:
                    X, Y, Z = x[:, :, idx], y[:, :, idx], z[:, :, idx]
                    slice_data = data[:, :, idx]

                # カラーマップの適用
                colors = cmap(norm(slice_data))

                # 断面の描画
                surf = ax.plot_surface(
                    X,
                    Y,
                    Z,
                    facecolors=colors,
                    alpha=opacity,
                    shade=False,
                )

                # 等高線の追加
                if kwargs.get("contour", False):
                    levels = kwargs.get("levels")
                    if levels is None:
                        levels = np.linspace(
                            np.nanmin(slice_data),
                            np.nanmax(slice_data),
                            kwargs.get("n_contours", 10),
                        )

                    # カラーが文字列の場合、RGBAに変換
                    colors = kwargs.get("contour_colors", "k")
                    if isinstance(colors, str):
                        colors = plt.cm.colors.to_rgba(colors)
                    elif isinstance(colors, (list, np.ndarray)):
                        # すでにRGBA形式であることを確認
                        if len(colors) not in [3, 4]:
                            raise ValueError(f"Invalid color format: {colors}")
                    else:
                        raise ValueError(f"Unsupported color type: {type(colors)}")

                    ax.contour(
                        X,
                        Y,
                        slice_data,
                        levels=levels,
                        colors=colors,
                        alpha=kwargs.get("contour_alpha", 0.5),
                        linewidths=kwargs.get("contour_linewidth", 1.0),
                    )

                # メタデータの更新
                metadata["slices"].append(
                    {
                        "axis": axis_name,
                        "position": float(pos),
                        "min": float(np.nanmin(slice_data)),
                        "max": float(np.nanmax(slice_data)),
                    }
                )

            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.warning(f"断面の生成エラー ({axis_name}): {e}")

    def _render_volume(
        self,
        ax: Axes3D,
        data: np.ndarray,
        cmap: plt.cm.ScalarMappable,
        norm: plt.Normalize,
        alpha: Optional[np.ndarray],
        metadata: Dict[str, Any],
    ) -> None:
        """ボリュームレンダリングを実行

        Args:
            ax: 3D Axes
            data: スカラー場データ
            cmap: カラーマップ
            norm: 正規化オブジェクト
            alpha: 透明度配列（オプション）
            metadata: 更新するメタデータ
        """
        # ボリュームレンダリングはまだ実験的な機能
        # 現在のmatplotlibではサポートが限定的
        metadata["display_type"].append("volume")

        try:
            # データのダウンサンプリング（必要に応じて）
            downsample = 2
            data_ds = data[::downsample, ::downsample, ::downsample]

            # カラーマップの適用
            colors = cmap(norm(data_ds))

            # 透明度の設定
            if alpha is None:
                # デフォルトの透明度関数
                alpha = np.clip(norm(data_ds), 0.1, 0.5)
            colors[..., 3] = alpha

            # ボクセルの描画
            nx, ny, nz = data_ds.shape
            positions = np.moveaxis(np.mgrid[0:nx, 0:ny, 0:nz], 0, -1) * downsample
            ax.voxels(
                positions[..., 0],
                positions[..., 1],
                positions[..., 2],
                data_ds > norm.vmin,
                facecolors=colors,
            )

            # メタデータの更新
            metadata["volume"] = {
                "downsample": downsample,
                "voxels": int(np.sum(data_ds > norm.vmin)),
            }

        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"ボリュームレンダリングエラー: {e}")
