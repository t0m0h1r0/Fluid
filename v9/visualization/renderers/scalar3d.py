"""3Dスカラー場の可視化を提供するモジュール

このモジュールは、3次元スカラー場の可視化機能を実装します。
Matplotlibを使用して等値面や断面の表示を行います。
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
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

    3次元スカラー場を可視化するためのレンダラークラスです。
    等値面表示や任意断面での可視化に対応します。
    """

    def __init__(self, config: VisualizationConfig):
        """3Dレンダラーを初期化"""
        super().__init__(config)

    def _compute_isosurface(
        self, data: np.ndarray, isovalue: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """等値面データを計算

        Args:
            data: 入力データ
            isovalue: 等値面の値

        Returns:
            頂点と面のタプル
        """
        if not HAVE_SKIMAGE:
            # scikit-imageがない場合は、単純な閾値処理による等値面を生成
            x, y, z = np.where((data >= isovalue) & (np.roll(data, 1) < isovalue))
            verts = np.column_stack([x, y, z])
            faces = np.arange(len(x)).reshape(-1, 3)
            return verts, faces
        else:
            # マーチングキューブ法で等値面を生成
            verts, faces, _, _ = measure.marching_cubes(data, isovalue)
            return verts, faces

    def render(
        self, data: np.ndarray, view: Optional[ViewConfig] = None, **kwargs
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """3Dスカラー場を描画

        Args:
            data: 描画するデータ
            view: 視点設定
            **kwargs: 追加の描画オプション
                - slice_positions: 各軸での断面位置のリスト
                - isovalues: 等値面の値のリスト
                - opacity: 不透明度
                - cmap: カラーマップ名

        Returns:
            (Figure, メタデータの辞書)
        """
        if data.ndim != 3:
            raise ValueError("3次元データが必要です")

        # 図の作成
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # データの正規化
        norm = self.create_normalizer(data)
        cmap = plt.get_cmap(kwargs.get("cmap", self.config.colormap))

        # グリッドの生成
        nx, ny, nz = data.shape
        x, y, z = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )

        # 等値面の描画
        isovalues = kwargs.get("isovalues", [])
        if isovalues:
            for isovalue in isovalues:
                try:
                    # 等値面を計算
                    verts, faces = self._compute_isosurface(data, isovalue)

                    # 等値面をポリゴンとして描画
                    mesh = Poly3DCollection(verts[faces])
                    color = cmap(norm(isovalue))
                    mesh.set_facecolor(color)
                    mesh.set_alpha(kwargs.get("opacity", 0.3))
                    ax.add_collection3d(mesh)
                except Exception as e:
                    print(f"等値面の生成エラー: {e}")

        # 断面の描画
        slice_positions = kwargs.get("slice_positions", [])
        if slice_positions:
            for axis, pos in enumerate(slice_positions):
                if pos is not None:
                    idx = int(pos * data.shape[axis])
                    if axis == 0:
                        X, Y, Z = x[idx, :, :], y[idx, :, :], z[idx, :, :]
                        C = data[idx, :, :]
                    elif axis == 1:
                        X, Y, Z = x[:, idx, :], y[:, idx, :], z[:, idx, :]
                        C = data[:, idx, :]
                    else:
                        X, Y, Z = x[:, :, idx], y[:, :, idx], z[:, :, idx]
                        C = data[:, :, idx]

                    # 断面をプロット
                    surf = ax.plot_surface(
                        X,
                        Y,
                        Z,
                        facecolors=cmap(norm(C)),
                        alpha=kwargs.get("opacity", 0.7),
                    )

        # 視点と軸の設定
        if view is not None:
            ax.view_init(elev=view.elevation, azim=view.azimuth)
            ax.dist = view.distance

        self.setup_3d_axes(ax, view, bounds=[(0, 0, 0), np.array(data.shape) - 1])

        # カラーバーの追加
        if self.config.show_colorbar:
            self.setup_colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax)

        # メタデータの収集
        metadata = {
            "data_range": {"min": float(norm.vmin), "max": float(norm.vmax)},
            "dimensions": data.shape,
            "view": {
                "elevation": view.elevation if view else None,
                "azimuth": view.azimuth if view else None,
                "distance": view.distance if view else None,
            },
        }

        if isovalues:
            metadata["isovalues"] = isovalues
        if slice_positions:
            metadata["slice_positions"] = slice_positions

        return fig, metadata
