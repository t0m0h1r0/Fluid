"""可視化のためのインターフェースと抽象クラス"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from .core.base import ViewConfig


class VisualizationStrategy(ABC):
    """可視化戦略の抽象基底クラス"""

    @abstractmethod
    def visualize(
        self,
        data: Any,
        name: str,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """データを可視化

        Args:
            data: 可視化するデータ
            name: 出力ファイル名のベース
            timestamp: 現在の時刻
            view: 視点設定
            **kwargs: 追加の可視化オプション

        Returns:
            生成された可視化ファイルのパス
        """
        pass


class MultiFieldVisualizationStrategy(VisualizationStrategy):
    """複数のフィールドを同時に可視化する戦略"""

    @abstractmethod
    def visualize_combined(
        self,
        fields: Dict[str, Any],
        name: str,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """複数のフィールドを組み合わせて可視化

        Args:
            fields: 可視化するフィールドの辞書
            name: 出力ファイル名のベース
            timestamp: 現在の時刻
            view: 視点設定
            **kwargs: 追加の可視化オプション

        Returns:
            生成された可視化ファイルのパス
        """
        pass


class VisualizationContext:
    """可視化のコンテキストを管理するクラス"""

    def __init__(self, strategy: VisualizationStrategy):
        """可視化戦略を設定

        Args:
            strategy: 使用する可視化戦略
        """
        self._strategy = strategy

    def set_strategy(self, strategy: VisualizationStrategy):
        """可視化戦略を動的に変更

        Args:
            strategy: 新しい可視化戦略
        """
        self._strategy = strategy

    def visualize(
        self,
        data: Any,
        name: str,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """データを可視化

        Args:
            data: 可視化するデータ
            name: 出力ファイル名のベース
            timestamp: 現在の時刻
            view: 視点設定
            **kwargs: 追加の可視化オプション

        Returns:
            生成された可視化ファイルのパス
        """
        return self._strategy.visualize(data, name, timestamp, view, **kwargs)

    def visualize_combined(
        self,
        fields: Dict[str, Any],
        name: str,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """複数のフィールドを組み合わせて可視化

        Args:
            fields: 可視化するフィールドの辞書
            name: 出力ファイル名のベース
            timestamp: 現在の時刻
            view: 視点設定
            **kwargs: 追加の可視化オプション

        Returns:
            生成された可視化ファイルのパス
        """
        if isinstance(self._strategy, MultiFieldVisualizationStrategy):
            return self._strategy.visualize_combined(
                fields, name, timestamp, view, **kwargs
            )
        raise NotImplementedError(
            "現在の可視化戦略は複数フィールドの可視化をサポートしていません"
        )


class VisualizationFactory:
    """可視化戦略を生成するファクトリクラス"""

    @staticmethod
    def create_strategy(
        strategy_type: str, config: Dict[str, Any]
    ) -> VisualizationStrategy:
        """指定された型の可視化戦略を生成

        Args:
            strategy_type: 可視化戦略の種類
            config: 設定辞書

        Returns:
            生成された可視化戦略
        """
        from .renderer_strategy import (
            Renderer2DVisualizationStrategy,
            Renderer3DVisualizationStrategy,
        )

        strategies = {
            "2d": Renderer2DVisualizationStrategy,
            "3d": Renderer3DVisualizationStrategy,
        }

        if strategy_type.lower() not in strategies:
            raise ValueError(f"未知の可視化戦略: {strategy_type}")

        return strategies[strategy_type.lower()](config)
