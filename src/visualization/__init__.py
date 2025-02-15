"""可視化システムのメインモジュール

このパッケージは、2次元・3次元の物理場の可視化機能を統合的に提供します。
"""

from .core.base import VisualizationConfig, ViewConfig
from .visualizer import Visualizer
from .interfaces import VisualizationFactory, VisualizationContext
from .renderer_strategy import (
    Renderer2DVisualizationStrategy,
    Renderer3DVisualizationStrategy,
)

# レンダラー
from .renderers import (
    Scalar2DRenderer,
    Scalar3DRenderer,
    Vector2DRenderer,
    Vector3DRenderer,
)

# エクスポーター
from .core.exporter import ImageExporter

# 可視化関数
from .multiview import visualize_simulation_state, create_multiview_visualization

__all__ = [
    # 設定関連
    "VisualizationConfig",
    "ViewConfig",
    # メインクラス
    "Visualizer",
    # インターフェースと戦略
    "VisualizationFactory",
    "VisualizationContext",
    "Renderer2DVisualizationStrategy",
    "Renderer3DVisualizationStrategy",
    # レンダラー
    "Scalar2DRenderer",
    "Scalar3DRenderer",
    "Vector2DRenderer",
    "Vector3DRenderer",
    # エクスポーター
    "ImageExporter",
    # 可視化関数
    "visualize_simulation_state",
    "create_multiview_visualization",
]
