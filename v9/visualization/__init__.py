"""可視化システムのメインモジュール

様々な物理量の可視化を提供します。
"""

from .config import VisualizationConfig
from .scalar import ScalarVisualizer
from .vector import VectorVisualizer
from .interface import InterfaceVisualizer
from .combined import CombinedVisualizer
from .state import StateVisualizer  # 追加


class Visualizer2D:
    """2D可視化システム

    様々な物理量の可視化を統合的に提供します。
    """

    def __init__(self, config=None):
        """可視化システムを初期化

        Args:
            config: 可視化設定（Noneの場合はデフォルト設定）
        """
        self.config = config or VisualizationConfig()

        # 各種可視化クラスを初期化
        self.scalar = ScalarVisualizer(self.config)
        self.vector = VectorVisualizer(self.config)
        self.interface = InterfaceVisualizer(self.config)
        self.combined = CombinedVisualizer(self.config)

    def visualize_scalar(self, data, name, timestamp, **kwargs):
        """スカラー場を可視化"""
        return self.scalar.visualize(data, name, timestamp, **kwargs)

    def visualize_vector(self, vector_components, name, timestamp, **kwargs):
        """ベクトル場を可視化"""
        return self.vector.visualize(vector_components, name, timestamp, **kwargs)

    def visualize_interface(self, levelset, timestamp, **kwargs):
        """界面を可視化"""
        return self.interface.visualize(levelset, timestamp, **kwargs)

    def visualize_combined(self, fields, timestamp, **kwargs):
        """複合フィールドを可視化"""
        return self.combined.visualize(fields, timestamp, **kwargs)

    @classmethod
    def from_config(cls, config_path):
        """設定ファイルから可視化システムを作成

        Args:
            config_path: 設定ファイルのパス

        Returns:
            初期化された可視化システム
        """
        config = VisualizationConfig.from_yaml(config_path)
        return cls(config)


# パッケージレベルでの関数を追加
def create_visualizer(config_path=None):
    """可視化システムを簡単に作成

    Args:
        config_path: オプションの設定ファイルパス

    Returns:
        Visualizer2Dインスタンス
    """
    if config_path:
        return Visualizer2D.from_config(config_path)
    return Visualizer2D()


# エクスポートする公開インターフェース
__all__ = [
    "Visualizer2D",
    "VisualizationConfig",
    "create_visualizer",
    "StateVisualizer",  # 追加
]
