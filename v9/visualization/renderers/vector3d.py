"""3Dベクトル場の可視化を提供するモジュール

このモジュールは、3次元ベクトル場の可視化機能を実装します。
ベクトル矢印表示、流線表示、断面表示などに対応します。
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from ..core.renderer import Renderer3D
from ..core.base import VisualizationConfig, ViewConfig


class Vector3DRenderer(Renderer3D):
    """3Dベクトル場のレンダラー
    
    ベクトル場を3D矢印や流線として表示します。
    """
    
    def __init__(self, config: VisualizationConfig):
        """3Dベクトルレンダラーを初期化"""
        super().__init__(config)

    def render(
        self,
        vector_components: List[np.ndarray],
        view: Optional[ViewConfig] = None,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Tuple[Figure, Dict[str, Any]]:
        """3Dベクトル場を描画

        Args:
            vector_components: ベクトル場の各成分 [u, v, w]
            view: 視点設定
            ax: 既存のAxes（Noneの場合は新規作成）
            **kwargs: 追加の描画オプション
                density: ベクトル矢印の密度
                scale: ベクトル矢印のスケール
                width: 矢印の幅
                headwidth: 矢印の頭の幅
                headlength: 矢印の頭の長さ
                color: ベクトルの色
                magnitude_colors: 大きさに応じた色付け
                alpha: 透明度
                streamlines: 流線の描画
                n_streamlines: 流線の数
                streamline_density: 流線の密度
                streamline_length: 流線の長さ
                slice_axes: 表示する断面の軸
                slice_positions: 各軸でのスライス位置

        Returns:
            (図, メタデータの辞書)のタプル
        """
        if len(vector_components) != 3:
            raise ValueError("3次元ベクトル場には3つの成分が必要です")

        u, v, w = vector_components
        if not all(comp.shape == u.shape for comp in [v, w]):
            raise ValueError("ベクトル成分の形状が一致しません")

        # 図とAxesの準備
        if ax is None:
            fig, ax = self.create_figure(projection="3d")
        else:
            fig = ax.figure
            if not isinstance(ax, Axes3D):
                raise ValueError("3D Axesが必要です")

        # ベクトルの大きさを計算
        magnitude = np.sqrt(u**2 + v**2 + w**2)
        norm = self.create_normalizer(magnitude, robust=True)

        # メタデータの初期化
        metadata = {
            "data_range": {
                "min_magnitude": float(np.min(magnitude)),
                "max_magnitude": float(np.max(magnitude))
            },
            "dimensions": u.shape,
            "display_type": []
        }

        # グリッドの生成
        nx, ny, nz = u.shape
        x, y, z = np.meshgrid(
            np.arange(nx),
            np.arange(ny),
            np.arange(nz),
            indexing="ij"
        )

        # ベクトル場の表示
        if not kwargs.get("streamlines_only", False):
            self._render_vectors(
                ax=ax,
                x=x, y=y, z=z,
                u=u, v=v, w=w,
                magnitude=magnitude,
                norm=norm,
                metadata=metadata,
                **kwargs
            )

        # 流線の描画
        if kwargs.get("streamlines", False):
            self._render_streamlines(
                ax=ax,
                u=u, v=v, w=w,
                magnitude=magnitude,
                norm=norm,
                metadata=metadata,
                **kwargs
            )

        # 断面の描画
        if view is not None and view.slice_axes:
            self._render_slices(
                ax=ax,
                x=x, y=y, z=z,
                u=u, v=v, w=w,
                magnitude=magnitude,
                norm=norm,
                view=view,
                metadata=metadata,
                **kwargs
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
        if kwargs.get("magnitude_colors", True) and self.config.show_colorbar:
            label = kwargs.get("colorbar_label", "Velocity magnitude")
            self.setup_colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis),
                ax,
                label=label
            )

        # タイトルの追加
        if kwargs.get("title"):
            ax.set_title(kwargs["title"])

        return fig, metadata

    def _render_vectors(
        self,
        ax: Axes3D,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        magnitude: np.ndarray,
        norm: plt.Normalize,
        metadata: Dict[str, Any],
        **kwargs
    ) -> None:
        """ベクトル矢印を描画

        Args:
            ax: 3D Axes
            x, y, z: 座標グリッド
            u, v, w: ベクトル成分
            magnitude: ベクトルの大きさ
            norm: 正規化オブジェクト
            metadata: 更新するメタデータ
            **kwargs: 描画オプション
        """
        metadata["display_type"].append("vectors")

        # 表示密度の設定
        density = kwargs.get("density", 15)
        scale = kwargs.get("scale", 1.0)
        skip = max(1, min(u.shape) // density)

        # ベクトル矢印の描画
        if kwargs.get("magnitude_colors", True):
            colors = plt.cm.viridis(norm(magnitude[::skip, ::skip, ::skip]))
        else:
            colors = kwargs.get("color", "k")

        # サンプリングしたデータでquiver
        ax.quiver(
            x[::skip, ::skip, ::skip],
            y[::skip, ::skip, ::skip],
            z[::skip, ::skip, ::skip],
            u[::skip, ::skip, ::skip],
            v[::skip, ::skip, ::skip],
            w[::skip, ::skip, ::skip],
            length=scale,
            normalize=True,
            colors=colors,
            alpha=kwargs.get("alpha", 1.0)
        )

        # メタデータの更新
        metadata["vectors"] = {
            "density": density,
            "scale": scale,
            "skip": skip
        }

    def _render_streamlines(
        self,
        ax: Axes3D,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        magnitude: np.ndarray,
        norm: plt.Normalize,
        metadata: Dict[str, Any],
        **kwargs
    ) -> None:
        """流線を描画

        Args:
            ax: 3D Axes
            u, v, w: ベクトル成分
            magnitude: ベクトルの大きさ
            norm: 正規化オブジェクト
            metadata: 更新するメタデータ
            **kwargs: 描画オプション
        """
        metadata["display_type"].append("streamlines")

        # 流線のパラメータ
        n_streamlines = kwargs.get("n_streamlines", 50)
        max_length = kwargs.get("streamline_length", 50)
        step_size = kwargs.get("step_size", 0.5)

        # 開始点をランダムに選択
        nx, ny, nz = u.shape
        start_points = np.random.rand(n_streamlines, 3) * np.array([nx, ny, nz])

        # 流線の計算と描画
        for start_point in start_points:
            points = self._compute_streamline(
                start_point=start_point,
                u=u, v=v, w=w,
                max_length=max_length,
                step_size=step_size
            )

            if len(points) > 1:
                points = np.array(points)
                # 流線の色を速度の大きさに基づいて設定
                speeds = np.sqrt(
                    np.gradient(points[:, 0])**2 +
                    np.gradient(points[:, 1])**2 +
                    np.gradient(points[:, 2])**2
                )
                colors = plt.cm.viridis(norm(speeds))

                # 流線をLine3DCollectionとして描画
                segments = np.stack([points[:-1], points[1:]], axis=1)
                lc = Line3DCollection(
                    segments,
                    colors=colors[:-1],
                    alpha=kwargs.get("streamline_alpha", 0.7),
                    linewidth=kwargs.get("streamline_width", 1)
                )
                ax.add_collection3d(lc)

        # メタデータの更新
        metadata["streamlines"] = {
            "count": n_streamlines,
            "max_length": max_length,
            "step_size": step_size
        }

    def _render_slices(
        self,
        ax: Axes3D,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        magnitude: np.ndarray,
        norm: plt.Normalize,
        view: ViewConfig,
        metadata: Dict[str, Any],
        **kwargs
    ) -> None:
        """断面上のベクトル場を描画

        Args:
            ax: 3D Axes
            x, y, z: 座標グリッド
            u, v, w: ベクトル成分
            magnitude: ベクトルの大きさ
            norm: 正規化オブジェクト
            view: 視点設定
            metadata: 更新するメタデータ
            **kwargs: 描画オプション
        """
        if "slices" not in metadata["display_type"]:
            metadata["display_type"].append("slices")
            metadata["slices"] = []

        # 各断面の描画
        axis_map = {"xy": 2, "yz": 0, "xz": 1}
        component_map = {
            "xy": ([0, 1], ["u", "v"]),
            "yz": ([1, 2], ["v", "w"]),
            "xz": ([0, 2], ["u", "w"])
        }

        for axis_name, pos in zip(view.slice_axes, view.slice_positions):
            try:
                axis = axis_map[axis_name.lower()]
                comp_indices, comp_names = component_map[axis_name.lower()]
                idx = int(pos * (u.shape[axis] - 1))

                # 座標とベクトル成分のスライスを取得
                if axis == 0:
                    X, Y, Z = x[idx, :, :], y[idx, :, :], z[idx, :, :]
                    components = [u[idx, :, :], v[idx, :, :], w[idx, :, :]]
                elif axis == 1:
                    X, Y, Z = x[:, idx, :], y[:, idx, :], z[:, idx, :]
                    components = [u[:, idx, :], v[:, idx, :], w[:, idx, :]]
                else:
                    X, Y, Z = x[:, :, idx], y[:, :, idx], z[:, :, idx]
                    components = [u[:, :, idx], v[:, :, idx], w[:, :, idx]]

                # 断面上のベクトル場を描画
                density = kwargs.get("slice_density", kwargs.get("density", 15))
                skip = max(1, min(X.shape) // density)

                if kwargs.get("magnitude_colors", True):
                    mag_slice = np.sqrt(sum(c**2 for c in components))
                    colors = plt.cm.viridis(norm(mag_slice[::skip, ::skip]))
                else:
                    colors = kwargs.get("color", "k")

                ax.quiver(
                    X[::skip, ::skip],
                    Y[::skip, ::skip],
                    Z[::skip, ::skip],
                    components[0][::skip, ::skip],
                    components[1][::skip, ::skip],
                    components[2][::skip, ::skip],
                    length=kwargs.get("scale", 1.0),
                    normalize=True,
                    colors=colors,
                    alpha=kwargs.get("alpha", 1.0)
                )

                # メタデータの更新
                metadata["slices"].append({
                    "axis": axis_name,
                    "position": float(pos),
                    "components": comp_names
                })

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"断面の生成エラー ({axis_name}): {e}")

    def _compute_streamline(
        self,
        start_point: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        max_length: int = 50,
        step_size: float = 0.5
    ) -> List[np.ndarray]:
        """流線を計算

        RK4法を用いて流線を数値積分します。

        Args:
            start_point: 開始点の座標
            u, v, w: ベクトル場の各成分
            max_length: 最大ステップ数
            step_size: 積分のステップサイズ

        Returns:
            流線上の点のリスト
        """
        points = [start_point.copy()]
        point = start_point.copy()

        # 速度場の補間関数
        def interpolate_velocity(pos):
            """位置での速度を補間"""
            # 範囲外チェック
            if (pos < 0).any() or (pos >= np.array(u.shape)).any():
                return None

            # 最近傍点での速度を返す
            ix, iy, iz = np.floor(pos).astype(int)
            try:
                return np.array([
                    u[ix, iy, iz],
                    v[ix, iy, iz],
                    w[ix, iy, iz]
                ])
            except IndexError:
                return None

        # RK4法による数値積分
        for _ in range(max_length):
            # 現在位置での速度（k1）
            k1 = interpolate_velocity(point)
            if k1 is None:
                break

            # 中間点での速度（k2）
            k2_pos = point + step_size * k1 / 2
            k2 = interpolate_velocity(k2_pos)
            if k2 is None:
                break

            # 中間点での速度（k3）
            k3_pos = point + step_size * k2 / 2
            k3 = interpolate_velocity(k3_pos)
            if k3 is None:
                break

            # 終点での速度（k4）
            k4_pos = point + step_size * k3
            k4 = interpolate_velocity(k4_pos)
            if k4 is None:
                break

            # 次の点を計算
            point = point + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            points.append(point.copy())

        return points

    def create_multiview(
        self,
        vector_components: List[np.ndarray],
        view: ViewConfig,
        **kwargs
    ) -> Dict[str, Tuple[Figure, Dict[str, Any]]]:
        """複数のビューを作成

        Args:
            vector_components: ベクトル場の各成分
            view: 視点設定
            **kwargs: 描画オプション

        Returns:
            ビュー名をキーとする (図, メタデータ) のタプルの辞書
        """
        result = {}

        # 3Dビュー
        fig_3d, meta_3d = self.render(
            vector_components,
            view=view,
            **kwargs
        )
        result["3D"] = (fig_3d, meta_3d)

        # 2D断面ビュー
        axis_map = {"xy": 2, "yz": 0, "xz": 1}
        component_map = {
            "xy": ([0, 1], ["u", "v"]),
            "yz": ([1, 2], ["v", "w"]),
            "xz": ([0, 2], ["u", "w"])
        }

        for axis_name, pos in zip(view.slice_axes, view.slice_positions):
            try:
                # 断面の設定を取得
                axis = axis_map[axis_name.lower()]
                comp_indices, comp_names = component_map[axis_name.lower()]

                # スライスの取得
                slice_components = [
                    self.create_slice(vector_components[i], axis, pos)
                    for i in comp_indices
                ]

                # スライス固有の設定を適用
                slice_kwargs = kwargs.copy()
                if "title" in slice_kwargs:
                    slice_kwargs["title"] = f"{slice_kwargs['title']} ({axis_name})"

                # 2Dレンダラーを使用してスライスを描画
                from .vector2d import Vector2DRenderer
                renderer_2d = Vector2DRenderer(self.config)
                fig_slice, meta_slice = renderer_2d.render(
                    slice_components,
                    **slice_kwargs
                )
                result[axis_name] = (fig_slice, meta_slice)

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"断面ビューの生成エラー ({axis_name}): {e}")

        return result

    def compute_derived_quantities(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        dx: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """ベクトル場の導出量を計算

        Args:
            u, v, w: ベクトル場の各成分
            dx: グリッド間隔

        Returns:
            導出量の辞書:
            - vorticity_x, vorticity_y, vorticity_z: 渦度の各成分
            - helicity: ヘリシティ
            - q_criterion: Qクライテリオン
            - divergence: 発散
        """
        # 勾配の計算
        dudy, dudz = np.gradient(u, dx, axis=(1, 2))
        dvdx, dvdz = np.gradient(v, dx, axis=(0, 2))
        dwdx, dwdy = np.gradient(w, dx, axis=(0, 1))

        # 渦度の計算 (ω = ∇ × v)
        vorticity_x = dwdy - dvdz
        vorticity_y = dudz - dwdx
        vorticity_z = dvdx - dudy

        # ヘリシティの計算 (H = v・ω)
        helicity = (
            u * vorticity_x +
            v * vorticity_y +
            w * vorticity_z
        )

        # Qクライテリオンの計算
        q_criterion = -0.5 * (
            dudy * dvdx + dudz * dwdx +
            dvdx * dudy + dvdz * dwdy +
            dwdx * dudz + dwdy * dvdz
        )

        # 発散の計算 (∇・v)
        divergence = (
            np.gradient(u, dx, axis=0) +
            np.gradient(v, dx, axis=1) +
            np.gradient(w, dx, axis=2)
        )

        return {
            "vorticity_x": vorticity_x,
            "vorticity_y": vorticity_y,
            "vorticity_z": vorticity_z,
            "helicity": helicity,
            "q_criterion": q_criterion,
            "divergence": divergence
        }