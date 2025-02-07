"""スカラー場の可視化を提供するモジュール

スカラー場（圧力、温度など）の可視化機能を実装します。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from .base import BaseVisualizer
from .config import VisualizationConfig
from .utils import prepare_2d_slice, compute_data_range


class ScalarVisualizer(BaseVisualizer):
    """スカラー場の可視化クラス"""

    def __init__(self, config: VisualizationConfig = None):
        """スカラー場可視化システムを初期化

        Args:
            config: 可視化設定
        """
        super().__init__(config)

    def visualize(self, data: np.ndarray, name: str, timestamp: float, **kwargs) -> str:
        """スカラー場を可視化

        Args:
            data: スカラー場のデータ
            name: 変数名
            timestamp: 時刻
            **kwargs: 追加の設定
                - slice_axis: スライスする軸（3Dデータ用）
                - slice_index: スライスのインデックス
                - symmetric: 対称な色範囲にするかどうか
                - colormap: カラーマップ
                - contour: 等高線を表示するかどうか

        Returns:
            保存された画像のファイルパス
        """
        # 2Dスライスを取得
        slice_axis = kwargs.get("slice_axis", 2)
        slice_index = kwargs.get("slice_index", None)
        data_2d = prepare_2d_slice(data, slice_axis, slice_index)

        # データの前処理: NaNとInfを除去
        data_2d = np.nan_to_num(data_2d, nan=0.0, posinf=0.0, neginf=0.0)

        # 図とAxesを作成
        fig, ax = self.create_figure()

        # カラーマップと色範囲の設定
        symmetric = kwargs.get("symmetric", name in ["pressure", "vorticity"])

        # データに有効な範囲があるか確認
        finite_data = data_2d[np.isfinite(data_2d)]
        if len(finite_data) == 0:
            # データが全て非有効な場合は単色の画像を生成
            im = ax.imshow(np.zeros_like(data_2d), cmap="gray")
            ax.set_title(f"{name} (t = {timestamp:.3f}s) - No Valid Data")
        else:
            # データの範囲を計算
            vmin, vmax = compute_data_range(finite_data, symmetric)

            # カラーマップを決定
            colormap = kwargs.get("colormap", self.config.colormap)
            cmap = cm.get_cmap(colormap)

            # 画像のプロット
            interpolation = self.config.scalar_plot.get("interpolation", "nearest")
            im = ax.imshow(
                data_2d.T,
                origin="lower",
                cmap=cmap,
                norm=Normalize(vmin=vmin, vmax=vmax),
                interpolation=interpolation,
            )

            # 等高線の追加（オプション）
            contour = kwargs.get(
                "contour", self.config.scalar_plot.get("contour", False)
            )
            if contour:
                # 等高線レベルを安全に生成
                n_levels = 10
                if vmin != vmax:
                    levels = np.linspace(vmin, vmax, n_levels)
                    try:
                        cs = ax.contour(data_2d.T, levels=levels, colors="k", alpha=0.5)
                        ax.clabel(cs, inline=True, fontsize=8)
                    except Exception as e:
                        print(f"等高線描画エラー: {e}")

            # カラーバーの追加
            if self.config.show_colorbar:
                plt.colorbar(im, ax=ax, label=name)

            # タイトルの設定
            ax.set_title(f"{name} (t = {timestamp:.3f}s)")

        # 図を保存
        return self.save_figure(fig, name, timestamp)
