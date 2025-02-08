"""可視化システムのメインモジュール

このパッケージは、2次元・3次元の物理場の可視化機能を統合的に提供します。
"""

import numpy as np

from .core.base import VisualizationConfig, ViewConfig
from .interfaces import (
    VisualizationStrategy,
    MultiFieldVisualizationStrategy,
    VisualizationContext,
    VisualizationFactory,
)
from .renderer_strategy import (
    Renderer2DVisualizationStrategy,
    Renderer3DVisualizationStrategy,
)
from .visualizer import Visualizer

__all__ = [
    "Visualizer",
    "VisualizationConfig",
    "ViewConfig",
    "VisualizationStrategy",
    "MultiFieldVisualizationStrategy",
    "VisualizationContext",
    "VisualizationFactory",
    "Renderer2DVisualizationStrategy",
    "Renderer3DVisualizationStrategy",
]


def visualize_simulation_state(state, config: dict, timestamp: float = 0.0) -> str:
    """シミュレーション状態を可視化する便利関数

    Args:
        state: シミュレーション状態
        config: 可視化設定
        timestamp: 現在の時刻

    Returns:
        生成された可視化ファイルのパス
    """
    # 状態からフィールドデータを抽出
    fields = {}

    # 速度場の抽出
    if hasattr(state, "velocity"):
        velocity_components = [
            _ensure_3d(comp.data) for comp in state.velocity.components
        ]
        fields["vector"] = velocity_components

    # 圧力場の抽出
    if hasattr(state, "pressure"):
        fields["pressure"] = _ensure_3d(state.pressure.data)

    # Level Set場の抽出
    if hasattr(state, "levelset"):
        fields["levelset"] = _ensure_3d(state.levelset.data)

    # 次元数に基づいて可視化戦略を選択
    ndim = len(state.velocity.shape) if hasattr(state, "velocity") else 3
    strategy_type = "3d" if ndim == 3 else "2d"

    # 可視化コンテキストの作成と可視化の実行
    strategy = VisualizationFactory.create_strategy(
        strategy_type, config.get("visualization", {})
    )
    visualization_context = VisualizationContext(strategy)

    return visualization_context.visualize_combined(
        fields,
        name=f"combined_{timestamp:.3f}",
        timestamp=timestamp,
    )


def _ensure_3d(data: np.ndarray) -> np.ndarray:
    """データを3次元配列に変換

    Args:
        data: 入力データ

    Returns:
        3次元配列に変換されたデータ
    """
    if data.ndim == 1:
        nx = int(np.sqrt(len(data)))
        return data.reshape(nx, nx, 1)
    elif data.ndim == 2:
        return data[..., np.newaxis]
    return data
