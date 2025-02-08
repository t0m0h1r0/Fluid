"""界面の可視化を提供するモジュール

Level Set法による界面の可視化機能を実装します。
"""

import numpy as np
import matplotlib.pyplot as plt

from .base import BaseVisualizer
from .config import VisualizationConfig
from .utils import prepare_2d_slice


class InterfaceVisualizer(BaseVisualizer):
    """界面の可視化クラス"""

    def __init__(self, config: VisualizationConfig = None):
        """界面可視化システムを初期化

        Args:
            config: 可視化設定
        """
        super().__init__(config)

    def visualize(self, levelset: np.ndarray, timestamp: float, **kwargs) -> str:
        """界面を可視化

        Args:
            levelset: Level Set関数のデータ
            timestamp: 時刻
            **kwargs: 追加の設定
                - slice_axis: スライスする軸（3Dデータ用）
                - slice_index: スライスのインデックス
                - filled: 領域を塗りつぶすかどうか
                - colors: 各相の色
                - extra_contours: 追加の等高線を表示するかどうか

        Returns:
            保存された画像のファイルパス
        """
        # 2Dスライスを取得
        slice_axis = kwargs.get("slice_axis", 2)
        slice_index = kwargs.get("slice_index", None)
        levelset_2d = prepare_2d_slice(levelset, slice_axis, slice_index)

        # データの前処理: NaNとInfを除去
        levelset_2d = np.nan_to_num(levelset_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # 図とAxesを作成
        fig, ax = self.create_figure()

        # 塗りつぶし設定
        filled = kwargs.get("filled", self.config.interface_plot.get("filled", True))

        # 色の設定
        colors = kwargs.get(
            "colors", self.config.interface_plot.get("colors", ["lightblue", "white"])
        )

        # 座標グリッドの設定
        x = np.arange(levelset_2d.shape[0])
        y = np.arange(levelset_2d.shape[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        # データの値の範囲をチェック
        vmin, vmax = np.min(levelset_2d), np.max(levelset_2d)
        if abs(vmax - vmin) < 1e-10:  # データがほぼ同じ値の場合
            # 単純な塗りつぶしで表示
            if vmin > 0:
                ax.fill_between(x, 0, levelset_2d.shape[1], color=colors[0], alpha=0.5)
            else:
                ax.fill_between(x, 0, levelset_2d.shape[1], color=colors[1], alpha=0.5)
        else:
            # 領域の塗りつぶし
            if filled:
                cs_filled = ax.contourf(
                    X, Y,
                    levelset_2d,
                    levels=[-np.inf, 0, np.inf],
                    colors=colors,
                    alpha=0.5,
                )

                # カラーバーの追加（値の範囲が十分にある場合のみ）
                if self.config.show_colorbar and abs(vmax - vmin) > 1e-6:
                    plt.colorbar(cs_filled, ax=ax, label="Phase")

            # 追加の等高線（オプション）
            if kwargs.get("extra_contours", False) and abs(vmax - vmin) > 1e-6:
                levels = np.linspace(vmin, vmax, 10)
                cs_extra = ax.contour(
                    X, Y,
                    levelset_2d,
                    levels=levels,
                    colors="gray",
                    alpha=0.3,
                    linestyles="--",
                )
                ax.clabel(cs_extra, inline=True, fontsize=8)

        # グリッドの範囲を設定
        ax.set_xlim(0, levelset_2d.shape[0])
        ax.set_ylim(0, levelset_2d.shape[1])

        # タイトルの設定
        ax.set_title(f"Interface (t = {timestamp:.3f}s)")

        # 図を保存
        return self.save_figure(fig, "interface", timestamp)