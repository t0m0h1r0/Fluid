"""複合フィールドの可視化を提供するモジュール

複数のフィールドを組み合わせて可視化します。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .base import BaseVisualizer
from .config import VisualizationConfig
from .utils import prepare_2d_slice, create_grid


class CombinedVisualizer(BaseVisualizer):
    """複合フィールドの可視化クラス"""

    def __init__(self, config: VisualizationConfig = None):
        """複合フィールド可視化システムを初期化

        Args:
            config: 可視化設定
        """
        super().__init__(config)

    def visualize(self, fields: dict, timestamp: float, **kwargs) -> str:
        """複数のフィールドを組み合わせて可視化

        Args:
            fields: フィールドの辞書
                - 'pressure': スカラー場（圧力）
                - 'velocity_x', 'velocity_y': ベクトル場の成分
                - 'levelset': Level Set関数
            timestamp: 時刻
            **kwargs: 追加の設定
                - slice_axis: スライスする軸（3Dデータ用）
                - slice_index: スライスのインデックス

        Returns:
            保存された画像のファイルパス
        """
        # スライス設定
        slice_axis = kwargs.get("slice_axis", 2)
        slice_index = kwargs.get("slice_index", None)

        # 図とAxesを作成（やや大きめのサイズ）
        fig, ax = self.create_figure(size=(10, 8))

        # 圧力場の表示
        if "pressure" in fields:
            pressure_2d = prepare_2d_slice(fields["pressure"], slice_axis, slice_index)
            vmin, vmax = np.min(pressure_2d), np.max(pressure_2d)
            im = ax.imshow(
                pressure_2d.T,
                origin="lower",
                cmap="coolwarm",
                alpha=0.7,
                norm=Normalize(vmin=vmin, vmax=vmax),
            )

            # カラーバーの追加
            if self.config.show_colorbar:
                plt.colorbar(im, ax=ax, label="Pressure")

        # 速度場の表示
        if all(k in fields for k in ["velocity_x", "velocity_y"]):
            # 2Dスライスを取得
            vx_2d = prepare_2d_slice(fields["velocity_x"], slice_axis, slice_index)
            vy_2d = prepare_2d_slice(fields["velocity_y"], slice_axis, slice_index)

            # グリッドの生成
            nx, ny = vx_2d.shape
            x, y = create_grid((nx, ny))

            # ベクトル密度の設定
            density = self.config.vector_plot.get("density", 20)
            skip = max(1, min(nx, ny) // density)

            # ベクトル場のプロット
            q = ax.quiver(
                x[::skip, ::skip],
                y[::skip, ::skip],
                vx_2d[::skip, ::skip],
                vy_2d[::skip, ::skip],
                color="k",
                alpha=0.7,
            )

            # スケールバーの追加
            ax.quiverkey(q, 0.9, 0.9, 1.0, r"1 m/s", labelpos="E", coordinates="figure")

        # 界面の表示
        if "levelset" in fields:
            levelset_2d = prepare_2d_slice(fields["levelset"], slice_axis, slice_index)
            ax.contour(levelset_2d.T, levels=[0], colors="k", linewidths=2)

        # タイトルの設定
        ax.set_title(f"Combined Fields (t = {timestamp:.3f}s)")

        # 図を保存
        return self.save_figure(fig, "combined", timestamp)
