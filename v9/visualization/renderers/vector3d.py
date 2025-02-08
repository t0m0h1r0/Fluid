"""3Dベクトル場の可視化を提供するモジュール

このモジュールは、3次元ベクトル場の可視化機能を実装します。
Matplotlibを使用して、ベクトル場の3D表示を行います。
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..core.base import Renderer, VisualizationConfig, ViewConfig


class Vector3DRenderer(Renderer):
    """3Dベクトル場のレンダラー

    3次元ベクトル場を可視化するためのレンダラークラスです。
    ベクトル矢印の表示や流線の描画に対応します。
    """

    def __init__(self, config: VisualizationConfig):
        """3Dレンダラーを初期化"""
        super().__init__(config)

    def render(
        self,
        vector_components: List[np.ndarray],
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, Dict[str, Any]]:
        """3Dベクトル場を描画

        Args:
            vector_components: ベクトル場の各成分のリスト [u, v, w]
            view: 視点設定
            **kwargs: 追加の描画オプション
                - density: ベクトル矢印の密度
                - scale: ベクトル矢印のスケール
                - slice_positions: 各軸での断面位置のリスト
                - streamlines: 流線を描画するかどうか
                - n_streamlines: 流線の数

        Returns:
            (Figure, メタデータの辞書)
        """
        if len(vector_components) != 3:
            raise ValueError("3次元ベクトル場には3つの成分が必要です")

        # データの取得
        u, v, w = vector_components

        # グリッドの生成
        x, y, z = np.meshgrid(
            np.arange(u.shape[0]),
            np.arange(u.shape[1]),
            np.arange(u.shape[2]),
            indexing="ij",
        )

        # 図の作成
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 視点の設定
        if view is None:
            view = ViewConfig()

        ax.view_init(elev=view.elevation, azim=view.azimuth)
        ax.dist = view.distance

        # ベクトル場の大きさを計算
        magnitude = np.sqrt(u**2 + v**2 + w**2)
        norm = Normalize(vmin=np.min(magnitude), vmax=np.max(magnitude))

        # ベクトル矢印の描画
        density = kwargs.get("density", 20)
        scale = kwargs.get("scale", 1.0)

        # サンプリング間隔の計算
        skip = max(1, min(u.shape) // density)

        # 断面でのベクトル場の描画
        slice_positions = kwargs.get("slice_positions", [])
        if slice_positions:
            for axis, pos in enumerate(slice_positions):
                if pos is not None:
                    idx = int(pos * u.shape[axis])
                    if axis == 0:
                        X, Y, Z = (
                            x[idx, ::skip, ::skip],
                            y[idx, ::skip, ::skip],
                            z[idx, ::skip, ::skip],
                        )
                        U, V, W = (
                            u[idx, ::skip, ::skip],
                            v[idx, ::skip, ::skip],
                            w[idx, ::skip, ::skip],
                        )
                    elif axis == 1:
                        X, Y, Z = (
                            x[::skip, idx, ::skip],
                            y[::skip, idx, ::skip],
                            z[::skip, idx, ::skip],
                        )
                        U, V, W = (
                            u[::skip, idx, ::skip],
                            v[::skip, idx, ::skip],
                            w[::skip, idx, ::skip],
                        )
                    else:
                        X, Y, Z = (
                            x[::skip, ::skip, idx],
                            y[::skip, ::skip, idx],
                            z[::skip, ::skip, idx],
                        )
                        U, V, W = (
                            u[::skip, ::skip, idx],
                            v[::skip, ::skip, idx],
                            w[::skip, ::skip, idx],
                        )

                    # ベクトル矢印の描画
                    ax.quiver(
                        X,
                        Y,
                        Z,
                        U,
                        V,
                        W,
                        length=scale,
                        normalize=True,
                        color=plt.cm.viridis(norm(np.sqrt(U**2 + V**2 + W**2))),
                    )

        # 流線の描画
        if kwargs.get("streamlines", False):
            n_streamlines = kwargs.get("n_streamlines", 50)

            # 開始点をランダムに選択
            start_points = np.random.rand(n_streamlines, 3) * np.array(u.shape)

            # 各開始点から流線を計算
            for start_point in start_points:
                points = self._compute_streamline(
                    start_point, (u, v, w), max_steps=100, step_size=0.1
                )
                if len(points) > 1:
                    points = np.array(points)
                    # 流線の色を速度の大きさに基づいて設定
                    colors = plt.cm.viridis(
                        norm(
                            np.sqrt(
                                np.gradient(points[:, 0]) ** 2
                                + np.gradient(points[:, 1]) ** 2
                                + np.gradient(points[:, 2]) ** 2
                            )
                        )
                    )
                    ax.plot(
                        points[:, 0],
                        points[:, 1],
                        points[:, 2],
                        color=colors[0],
                        alpha=0.5,
                    )

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
            plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=self.config.colormap),
                ax=ax,
                label="Velocity magnitude",
            )

        # メタデータの収集
        metadata = {
            "view": {
                "elevation": view.elevation,
                "azimuth": view.azimuth,
                "distance": view.distance,
            },
            "data_range": {
                "min_magnitude": float(np.min(magnitude)),
                "max_magnitude": float(np.max(magnitude)),
            },
        }

        return fig, metadata

    def _compute_streamline(
        self,
        start_point: np.ndarray,
        velocity_field: Tuple[np.ndarray, np.ndarray, np.ndarray],
        max_steps: int = 100,
        step_size: float = 0.1,
    ) -> List[np.ndarray]:
        """流線を計算

        RK4法を用いて流線を計算します。

        Args:
            start_point: 開始点の座標
            velocity_field: 速度場の各成分 (u, v, w)
            max_steps: 最大ステップ数
            step_size: 積分のステップサイズ

        Returns:
            流線上の点のリスト
        """
        u, v, w = velocity_field
        points = [start_point]
        point = start_point.copy()

        def interpolate_velocity(pos):
            """速度場を補間"""
            if (pos < 0).any() or (pos >= np.array(u.shape)).any():
                return None

            # 最近傍点での速度を返す（より高度な補間も可能）
            ix, iy, iz = np.floor(pos).astype(int)
            return np.array([u[ix, iy, iz], v[ix, iy, iz], w[ix, iy, iz]])

        for _ in range(max_steps):
            # RK4法による積分
            k1 = interpolate_velocity(point)
            if k1 is None:
                break

            k2_pos = point + step_size * k1 / 2
            k2 = interpolate_velocity(k2_pos)
            if k2 is None:
                break

            k3_pos = point + step_size * k2 / 2
            k3 = interpolate_velocity(k3_pos)
            if k3 is None:
                break

            k4_pos = point + step_size * k3
            k4 = interpolate_velocity(k4_pos)
            if k4 is None:
                break

            # 次の点を計算
            point = point + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            points.append(point.copy())

        return points
