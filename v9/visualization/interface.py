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

        # データの範囲を確認
        if not np.isfinite(levelset_2d).any():
            # データが全て非有効な場合は単色の画像を生成
            ax.imshow(np.zeros_like(levelset_2d), cmap="gray")
            ax.set_title(f"Interface (t = {timestamp:.3f}s) - No Valid Data")
        else:
            # 領域の塗りつぶし
            if filled:
                # 有効な等高線レベルを見つける
                unique_levels = np.unique(levelset_2d)
                valid_levels = unique_levels[np.isfinite(unique_levels)]

                if len(valid_levels) > 1:
                    # 有効な等高線レベルが2つ以上ある場合
                    try:
                        # 等高線で塗りつぶし
                        cs_filled = ax.contourf(
                            levelset_2d.T,
                            levels=[-np.inf, 0, np.inf],
                            colors=colors,
                            alpha=0.5,
                        )

                        # カラーバーの追加を試みる
                        if self.config.show_colorbar:
                            # カラーバーのマッピングを手動で作成
                            plt.colorbar(cs_filled, ax=ax, label="Phase")

                    except Exception as e:
                        # 等高線描画に失敗した場合の代替表示
                        print(f"等高線描画エラー: {e}")
                        ax.imshow(levelset_2d.T, cmap="coolwarm", alpha=0.5)

            # 追加の等高線（オプション）
            extra_contours = kwargs.get("extra_contours", False)
            if extra_contours:
                try:
                    # 有効な等高線レベルを計算
                    levels = np.linspace(
                        np.nanmin(levelset_2d), np.nanmax(levelset_2d), 10
                    )
                    levels = levels[np.isfinite(levels)]

                    if len(levels) > 1:
                        cs_extra = ax.contour(
                            levelset_2d.T,
                            levels=levels,
                            colors="gray",
                            alpha=0.3,
                            linestyles="--",
                        )
                        ax.clabel(cs_extra, inline=True, fontsize=8)
                except Exception as e:
                    print(f"追加等高線描画エラー: {e}")

            # タイトルの設定
            ax.set_title(f"Interface (t = {timestamp:.3f}s)")

        # 図を保存
        return self.save_figure(fig, "interface", timestamp)
