"""可視化戦略の実装"""

from typing import Any, Optional, Dict
from .core.base import ViewConfig
from .interfaces import MultiFieldVisualizationStrategy
from .visualizer import Visualizer


class Renderer2DVisualizationStrategy(MultiFieldVisualizationStrategy):
    """2Dレンダラーを使用した可視化戦略"""

    def __init__(self, config: Dict[str, Any]):
        """戦略を初期化

        Args:
            config: 可視化設定
        """
        self.visualizer = Visualizer(config)

    def visualize(
        self,
        data: Any,
        name: str,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """2Dデータを可視化

        Args:
            data: 可視化するデータ
            name: 出力ファイル名のベース
            timestamp: 現在の時刻
            view: 視点設定
            **kwargs: 追加の可視化オプション

        Returns:
            生成された可視化ファイルのパス
        """
        if isinstance(data, list):  # ベクトル場
            return self.visualizer.visualize_vector(
                data, name, timestamp, view, **kwargs
            )
        else:  # スカラー場
            return self.visualizer.visualize_scalar(
                data, name, timestamp, view, **kwargs
            )

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
        # マップされたフィールドを準備
        scalar_data = fields.get("scalar") or fields.get("pressure")
        vector_data = fields.get("vector")

        return self.visualizer.visualize_combined(
            scalar_data=scalar_data,
            vector_components=vector_data,
            name=name,
            timestamp=timestamp,
            view=view,
            **kwargs,
        )


class Renderer3DVisualizationStrategy(MultiFieldVisualizationStrategy):
    """3Dレンダラーを使用した可視化戦略"""

    def __init__(self, config: Dict[str, Any]):
        """戦略を初期化

        Args:
            config: 可視化設定
        """
        self.visualizer = Visualizer(config)

    def visualize(
        self,
        data: Any,
        name: str,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """3Dデータを可視化

        Args:
            data: 可視化するデータ
            name: 出力ファイル名のベース
            timestamp: 現在の時刻
            view: 視点設定
            **kwargs: 追加の可視化オプション

        Returns:
            生成された可視化ファイルのパス
        """
        if isinstance(data, list):  # ベクトル場
            return self.visualizer.visualize_vector(
                data, name, timestamp, view, **kwargs
            )
        else:  # スカラー場
            return self.visualizer.visualize_scalar(
                data, name, timestamp, view, **kwargs
            )

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
        # マップされたフィールドを準備
        scalar_data = fields.get("scalar") or fields.get("pressure")
        vector_data = fields.get("vector")
        levelset_data = fields.get("levelset")

        # レベルセットを追加の可視化オプションに含める
        if levelset_data is not None:
            if "scalar_options" not in kwargs:
                kwargs["scalar_options"] = {}
            kwargs["scalar_options"]["levelset"] = levelset_data

        return self.visualizer.visualize_combined(
            scalar_data=scalar_data,
            vector_components=vector_data,
            name=name,
            timestamp=timestamp,
            view=view,
            **kwargs,
        )
