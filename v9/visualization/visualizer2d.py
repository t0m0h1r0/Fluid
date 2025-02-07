"""2D可視化システムを提供するモジュール

このモジュールは、2D可視化のための実装を提供します。
matplotlibを使用して、スカラー場、ベクトル場、界面などを可視化します。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Dict, Any, Optional, List, Tuple
from .base import Visualizer, VisualizationConfig


class Visualizer2D(Visualizer):
    """2D可視化クラス

    matplotlibを使用して2D可視化を行います。
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """2D可視化システムを初期化"""
        super().__init__(config)
        plt.style.use("default")

    def create_figure(
        self, size: Tuple[float, float] = (8, 6), projection: Optional[str] = None
    ) -> Tuple[Any, Any]:
        """図とAxesを作成"""
        fig, ax = plt.subplots(figsize=size)
        self.apply_common_settings(ax)
        return fig, ax

    def visualize_scalar(
        self, data: np.ndarray, name: str, timestamp: float, **kwargs
    ) -> None:
        """スカラー場を可視化

        Args:
            data: スカラー場のデータ
            name: 変数名
            timestamp: 時刻
            **kwargs: 追加の設定
                - slice_axis: スライスする軸（3D data用）
                - slice_index: スライスのインデックス
                - contour: 等高線を描画するかどうか
        """
        # 3Dデータの場合はスライスを取得
        if data.ndim == 3:
            slice_axis = kwargs.get("slice_axis", 2)  # デフォルトはz軸
            slice_index = kwargs.get("slice_index", data.shape[slice_axis] // 2)
            slices = [slice(None)] * 3
            slices[slice_axis] = slice_index
            data = data[tuple(slices)]

        fig, ax = self.create_figure()

        # スケールと範囲の設定
        scale, (vmin, vmax) = self.get_scale_and_range(
            data, symmetric=name in ["pressure", "vorticity"]
        )

        # カラーマップでプロット
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=self.config.colormap,
            norm=Normalize(vmin=vmin, vmax=vmax),
            interpolation="nearest",
        )

        # 等高線の追加（オプション）
        if kwargs.get("contour", False):
            levels = np.linspace(vmin, vmax, 10)
            cs = ax.contour(data.T, levels=levels, colors="k", alpha=0.5)
            ax.clabel(cs, inline=True, fontsize=8)

        # カラーバーの追加
        if self.config.show_colorbar:
            self.add_colorbar(im, name)

        # タイトルの設定
        ax.set_title(f"{name} (t = {timestamp:.3f}s)")

        # 保存
        plt.savefig(
            self.create_filename(name, timestamp),
            dpi=self.config.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def visualize_vector(
        self, data: List[np.ndarray], name: str, timestamp: float, **kwargs
    ) -> None:
        """ベクトル場を可視化

        Args:
            data: ベクトル場の各成分のデータ
            name: 変数名
            timestamp: 時刻
            **kwargs: 追加の設定
                - scale: ベクトルのスケール
                - density: ベクトルの密度
                - magnitude: 大きさも表示するかどうか
        """
        if len(data) < 2:
            raise ValueError("2D可視化には少なくとも2つの成分が必要です")

        # 3Dデータの場合はスライスを取得
        if data[0].ndim == 3:
            slice_axis = kwargs.get("slice_axis", 2)
            slice_index = kwargs.get("slice_index", data[0].shape[slice_axis] // 2)
            slices = [slice(None)] * 3
            slices[slice_axis] = slice_index
            data = [d[tuple(slices)] for d in data[:2]]  # x, y成分のみ使用

        fig, ax = self.create_figure()

        # ベクトル場の格子点を生成
        nx, ny = data[0].shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

        # ベクトルの密度を調整
        density = kwargs.get("density", 20)
        skip = max(1, min(nx, ny) // density)

        # ベクトル場をプロット
        scale = kwargs.get("scale", None)
        q = ax.quiver(
            x[::skip, ::skip],
            y[::skip, ::skip],
            data[0][::skip, ::skip],
            data[1][::skip, ::skip],
            scale=scale,
        )

        # ベクトルの大きさもプロット（オプション）
        if kwargs.get("magnitude", True):
            magnitude = np.sqrt(data[0] ** 2 + data[1] ** 2)
            im = ax.imshow(
                magnitude.T, origin="lower", cmap=self.config.colormap, alpha=0.3
            )
            if self.config.show_colorbar:
                self.add_colorbar(im, f"{name} magnitude")

        # スケールバーの追加
        ax.quiverkey(q, 0.9, 0.9, 1.0, r"1 m/s", labelpos="E", coordinates="figure")

        # タイトルの設定
        ax.set_title(f"{name} (t = {timestamp:.3f}s)")

        # 保存
        plt.savefig(
            self.create_filename(name, timestamp),
            dpi=self.config.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def visualize_interface(
        self, levelset: np.ndarray, timestamp: float, **kwargs
    ) -> None:
        """界面を可視化

        Args:
            levelset: Level Set関数のデータ
            timestamp: 時刻
            **kwargs: 追加の設定
                - filled: 領域を塗りつぶすかどうか
                - colors: 各相の色
        """
        # 3Dデータの場合はスライスを取得
        if levelset.ndim == 3:
            slice_axis = kwargs.get("slice_axis", 2)
            slice_index = kwargs.get("slice_index", levelset.shape[slice_axis] // 2)
            slices = [slice(None)] * 3
            slices[slice_axis] = slice_index
            levelset = levelset[tuple(slices)]

        fig, ax = self.create_figure()

        # 塗りつぶしオプション
        if kwargs.get("filled", True):
            colors = kwargs.get("colors", ["lightblue", "white"])
            cs_filled = ax.contourf(
                levelset.T, levels=[-np.inf, 0, np.inf], colors=colors, alpha=0.5
            )

            if self.config.show_colorbar:
                plt.colorbar(cs_filled, ax=ax, label="Phase")

        # 追加の等高線（オプション）
        if kwargs.get("extra_contours", False):
            levels = np.linspace(np.min(levelset), np.max(levelset), 10)
            cs_extra = ax.contour(
                levelset.T, levels=levels, colors="gray", alpha=0.3, linestyles="--"
            )
            ax.clabel(cs_extra, inline=True, fontsize=8)

        # タイトルの設定
        ax.set_title(f"Interface (t = {timestamp:.3f}s)")

        # 保存
        plt.savefig(
            self.create_filename("interface", timestamp),
            dpi=self.config.dpi,
            bbox_inches="tight",
        )
        plt.close()

    def add_colorbar(
        self, mappable: Any, label: str, orientation: str = "vertical"
    ) -> None:
        """カラーバーを追加"""
        plt.colorbar(mappable, label=label, orientation=orientation)

    def visualize_combined(
        self, fields: Dict[str, np.ndarray], timestamp: float, **kwargs
    ) -> None:
        """複数のフィールドを組み合わせて可視化

        Args:
            fields: フィールドのディクショナリ
            timestamp: 時刻
            **kwargs: 追加の設定
        """
        fig, ax = self.create_figure(size=(10, 8))

        # スカラー場（圧力など）の表示
        if "pressure" in fields:
            scale, (vmin, vmax) = self.get_scale_and_range(
                fields["pressure"], symmetric=True
            )
            im = ax.imshow(
                fields["pressure"].T,
                origin="lower",
                cmap="coolwarm",
                alpha=0.7,
                norm=Normalize(vmin=vmin, vmax=vmax),
            )
            if self.config.show_colorbar:
                self.add_colorbar(im, "Pressure")

        # 速度場の表示
        if all(k in fields for k in ["velocity_x", "velocity_y"]):
            # ベクトル場の格子点とダウンサンプリング
            nx, ny = fields["velocity_x"].shape
            x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
            skip = max(1, min(nx, ny) // 20)

            # ベクトル場のプロット
            q = ax.quiver(
                x[::skip, ::skip],
                y[::skip, ::skip],
                fields["velocity_x"][::skip, ::skip],
                fields["velocity_y"][::skip, ::skip],
                color="k",
                alpha=0.7,
            )

            # スケールバーの追加
            ax.quiverkey(q, 0.9, 0.9, 1.0, r"1 m/s", labelpos="E", coordinates="figure")

        # 界面の表示
        if "levelset" in fields:
            cs = ax.contour(fields["levelset"].T, levels=[0], colors="k", linewidths=2)

        # タイトルの設定
        ax.set_title(f"Combined Fields (t = {timestamp:.3f}s)")

        # 保存
        plt.savefig(
            self.create_filename("combined", timestamp),
            dpi=self.config.dpi,
            bbox_inches="tight",
        )
        plt.close()
