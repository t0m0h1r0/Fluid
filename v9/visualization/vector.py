"""ベクトル場の可視化を提供するモジュール

ベクトル場（速度場など）の可視化機能を実装します。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .base import BaseVisualizer
from .config import VisualizationConfig
from .utils import (
    prepare_2d_slice, 
    compute_vector_magnitude,
    create_grid
)


class VectorVisualizer(BaseVisualizer):
    """ベクトル場の可視化クラス"""

    def __init__(self, config: VisualizationConfig = None):
        """ベクトル場可視化システムを初期化

        Args:
            config: 可視化設定
        """
        super().__init__(config)

    def visualize(
        self, 
        vector_components: list, 
        name: str, 
        timestamp: float, 
        **kwargs
    ) -> str:
        """ベクトル場を可視化

        Args:
            vector_components: ベクトルの各成分のデータリスト
            name: 変数名
            timestamp: 時刻
            **kwargs: 追加の設定
                - slice_axis: スライスする軸（3Dデータ用）
                - slice_index: スライスのインデックス
                - density: ベクトル矢印の密度
                - scale: ベクトル矢印のスケール
                - magnitude: 大きさも表示するかどうか

        Returns:
            保存された画像のファイルパス
        """
        if len(vector_components) < 2:
            raise ValueError("2D可視化には少なくとも2つの成分が必要です")

        # 各成分の2Dスライスを取得
        slice_axis = kwargs.get('slice_axis', 2)
        slice_index = kwargs.get('slice_index', None)
        components_2d = [
            prepare_2d_slice(comp, slice_axis, slice_index) 
            for comp in vector_components[:2]
        ]

        # 図とAxesを作成
        fig, ax = self.create_figure()

        # ベクトル場の格子点を生成
        nx, ny = components_2d[0].shape
        x, y = create_grid((nx, ny))

        # ベクトルの密度を調整
        density = self.config.vector_plot.get('density', 20)
        skip = max(1, min(nx, ny) // density)
        
        # ベクトル矢印のスケール
        scale = kwargs.get('scale', self.config.vector_plot.get('scale', None))

        # ベクトル場をプロット
        q = ax.quiver(
            x[::skip, ::skip],
            y[::skip, ::skip],
            components_2d[0][::skip, ::skip],
            components_2d[1][::skip, ::skip],
            scale=scale,
        )

        # ベクトルの大きさも表示（オプション）
        magnitude_display = kwargs.get(
            'magnitude', 
            True  # デフォルトは大きさを表示
        )
        if magnitude_display:
            # ベクトルの大きさを計算
            magnitude = compute_vector_magnitude(components_2d)
            
            # 大きさを画像として表示
            im = ax.imshow(
                magnitude.T, 
                origin='lower', 
                cmap=self.config.colormap, 
                alpha=0.3
            )

            # カラーバーの追加
            if self.config.show_colorbar:
                plt.colorbar(im, ax=ax, label=f"{name} magnitude")

        # スケールバーの追加
        ax.quiverkey(
            q, 0.9, 0.9, 1.0, 
            r"1 m/s", 
            labelpos="E", 
            coordinates="figure"
        )

        # タイトルの設定
        ax.set_title(f"{name} (t = {timestamp:.3f}s)")

        # 図を保存
        return self.save_figure(fig, name, timestamp)
