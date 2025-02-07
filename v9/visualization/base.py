"""可視化システムの基底クラスを提供するモジュール

可視化の基本的な機能を定義します。
"""

import matplotlib.pyplot as plt
from typing import Tuple
from .config import VisualizationConfig


class BaseVisualizer:
    """可視化の基底クラス

    共通の可視化機能を提供します。
    """

    def __init__(self, config: VisualizationConfig = None):
        """可視化システムを初期化

        Args:
            config: 可視化設定（指定しない場合はデフォルト値）
        """
        # デフォルト設定を使用
        self.config = config or VisualizationConfig()
        
        # 出力ディレクトリを作成
        self.config.ensure_output_dir()
        
        # デフォルトのプロットスタイル
        plt.style.use("default")

    def create_figure(
        self, 
        size: Tuple[float, float] = (8, 6), 
        projection: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """図とAxesを作成

        Args:
            size: 図のサイズ
            projection: プロジェクションの種類

        Returns:
            (図, Axes)のタプル
        """
        fig, ax = plt.subplots(figsize=size)
        
        # 背景色の設定
        ax.set_facecolor('white')

        # 軸の設定
        if self.config.show_axes:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            ax.set_axis_off()

        # グリッドの設定
        if self.config.show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)

        return fig, ax

    def save_figure(
        self, 
        fig: plt.Figure, 
        prefix: str, 
        timestamp: float
    ) -> str:
        """図を保存

        Args:
            fig: 保存する図
            prefix: ファイル名のプレフィックス
            timestamp: タイムスタンプ

        Returns:
            保存されたファイルパス
        """
        # ファイル名を生成
        filename = self.config.get_output_filename(prefix, timestamp)
        
        # 図を保存
        fig.savefig(
            filename, 
            dpi=self.config.dpi, 
            bbox_inches="tight"
        )
        
        # figureを閉じる
        plt.close(fig)

        return filename
