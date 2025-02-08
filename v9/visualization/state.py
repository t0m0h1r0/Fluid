"""シミュレーション状態の可視化モジュール

このモジュールは、シミュレーション状態の可視化を管理するクラスを提供します。
"""

from typing import Dict, Any, Optional
import logging
import numpy as np

from visualization.interfaces import VisualizationContext, VisualizationFactory
from .core.base import ViewConfig


class StateVisualizer:
    """シミュレーション状態の可視化クラス

    シミュレーション状態の2D/3D可視化を統合的に管理します。
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """シミュレーション状態の可視化を初期化

        Args:
            config: 設定辞書
            logger: ロガー（オプション）
        """
        # 可視化戦略の選択（2Dか3D）
        ndim = len(config.get("domain", {}).get("dimensions", [64, 64, 64]))
        strategy_type = "3d" if ndim == 3 else "2d"

        # 可視化コンテキストの初期化
        visualization_strategy = VisualizationFactory.create_strategy(
            strategy_type, config.get("visualization", {})
        )
        self.visualization_context = VisualizationContext(visualization_strategy)

        self.logger = logger or logging.getLogger(__name__)
        self.config = config

    def _extract_field_data(self, state) -> Dict[str, Any]:
        """状態からフィールドデータを抽出

        Args:
            state: シミュレーション状態

        Returns:
            フィールドデータの辞書
        """
        fields = {}

        # 速度場の抽出
        if hasattr(state, "velocity"):
            try:
                velocity_components = [
                    self._ensure_3d(comp.data) for comp in state.velocity.components
                ]
                fields["vector"] = velocity_components
            except Exception as e:
                self.logger.warning(f"速度場の抽出中にエラー: {e}")

        # 圧力場の抽出
        if hasattr(state, "pressure"):
            try:
                fields["pressure"] = self._ensure_3d(state.pressure.data)
            except Exception as e:
                self.logger.warning(f"圧力場の抽出中にエラー: {e}")

        # Level Set場の抽出
        if hasattr(state, "levelset"):
            try:
                fields["levelset"] = self._ensure_3d(state.levelset.data)
            except Exception as e:
                self.logger.warning(f"Level Set場の抽出中にエラー: {e}")

        return fields

    def _ensure_3d(self, data: np.ndarray) -> np.ndarray:
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

    def visualize(
        self,
        state,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
    ) -> str:
        """状態を可視化

        Args:
            state: シミュレーション状態
            timestamp: 現在の時刻
            view: 視点設定

        Returns:
            生成された可視化ファイルのパス
        """
        try:
            # デフォルトのビュー設定
            if view is None:
                view = ViewConfig()

            # フィールドデータの抽出
            fields = self._extract_field_data(state)

            # 個別のフィールドを可視化
            for field_name, field_data in fields.items():
                self._visualize_single_field(field_data, field_name, timestamp, view)

            # 複数フィールドの組み合わせ可視化
            result = self.visualization_context.visualize_combined(
                fields,
                name=f"combined_{timestamp:.3f}",
                timestamp=timestamp,
                view=view,
            )

            return result

        except Exception as e:
            if self.logger:
                self.logger.error(f"可視化中にエラーが発生: {str(e)}")
            raise

    def _visualize_single_field(
        self,
        field_data: Any,
        field_name: str,
        timestamp: float,
        view: Optional[ViewConfig] = None,
    ) -> str:
        """単一フィールドを可視化

        Args:
            field_data: 可視化するフィールドデータ
            field_name: フィールドの名前
            timestamp: 現在の時刻
            view: 視点設定

        Returns:
            生成された可視化ファイルのパス
        """
        try:
            # フィールドデータの可視化
            return self.visualization_context.visualize(
                field_data,
                name=f"{field_name}_{timestamp:.3f}",
                timestamp=timestamp,
                view=view,
                # 追加のフィールド固有のオプションを設定可能
                contour=True,
                levels=[0] if field_name == "levelset" else None,
                interface_color="black" if field_name == "levelset" else None,
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"{field_name}フィールドの可視化中にエラー: {e}")
            return ""
