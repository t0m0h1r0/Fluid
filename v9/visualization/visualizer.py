"""可視化システムのメインクラスを提供するモジュール

このモジュールは、2次元・3次元の物理場の可視化機能を統合的に提供します。
"""

from typing import Optional, Dict, Any, Union, List

import numpy as np
from matplotlib import pyplot as plt

from .core.base import VisualizationConfig, ViewConfig
from .core.exporter import ImageExporter
from .renderers.scalar2d import Scalar2DRenderer
from .renderers.vector2d import Vector2DRenderer
from .renderers.scalar3d import Scalar3DRenderer
from .renderers.vector3d import Vector3DRenderer


class Visualizer:
    """可視化システムのメインクラス

    2次元・3次元の物理場の可視化を統合的に管理します。
    """

    def __init__(self, config: Union[VisualizationConfig, Dict[str, Any]]):
        """可視化システムを初期化

        Args:
            config: 設定（VisualizationConfigまたは辞書）
        """
        if isinstance(config, dict):
            self.config = VisualizationConfig.from_dict(config)
        else:
            self.config = config

        self.exporter = ImageExporter(self.config)

        # レンダラーの初期化
        self._renderers = {
            "2D": {
                "scalar": Scalar2DRenderer(self.config),
                "vector": Vector2DRenderer(self.config),
            },
            "3D": {
                "scalar": Scalar3DRenderer(self.config),
                "vector": Vector3DRenderer(self.config),
            },
        }

    def visualize_scalar(
        self,
        data: np.ndarray,
        name: str,
        timestamp: Optional[float] = None,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """スカラー場を可視化"""
        # データの次元に応じてレンダラーを選択
        renderer = self._renderers["3D" if data.ndim == 3 else "2D"]["scalar"]

        # 描画を実行
        fig, metadata = renderer.render(data, view, **kwargs)

        # 出力パスの生成
        filepath = self.config.get_output_path(name, timestamp)

        # 描画結果の出力
        self.exporter.export(fig, filepath, metadata)

        return str(filepath)

    def visualize_vector(
        self,
        vector_components: List[np.ndarray],
        name: str,
        timestamp: Optional[float] = None,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """ベクトル場を可視化"""
        # データの次元に応じてレンダラーを選択
        ndim = len(vector_components)
        renderer = self._renderers["3D" if ndim == 3 else "2D"]["vector"]

        # 描画を実行
        fig, metadata = renderer.render(vector_components, view, **kwargs)

        # 出力パスの生成
        filepath = self.config.get_output_path(name, timestamp)

        # 描画結果の出力
        self.exporter.export(fig, filepath, metadata)

        return str(filepath)

    def visualize_combined(
        self,
        scalar_data: Optional[np.ndarray] = None,
        vector_components: Optional[List[np.ndarray]] = None,
        name: str = "combined",
        timestamp: Optional[float] = None,
        view: Optional[ViewConfig] = None,
        **kwargs,
    ) -> str:
        """スカラー場とベクトル場を重ねて可視化"""
        if scalar_data is not None:
            ndim = scalar_data.ndim
        elif vector_components is not None:
            ndim = len(vector_components)
        else:
            raise ValueError("スカラー場またはベクトル場が必要です")

        # 図の作成
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d" if ndim == 3 else None)

        metadata = {}

        # スカラー場の描画
        if scalar_data is not None:
            renderer = self._renderers["3D" if ndim == 3 else "2D"]["scalar"]
            _, scalar_metadata = renderer.render(
                scalar_data, view, ax=ax, **kwargs.get("scalar_options", {})
            )
            metadata["scalar"] = scalar_metadata

        # ベクトル場の描画
        if vector_components is not None:
            renderer = self._renderers["3D" if ndim == 3 else "2D"]["vector"]
            _, vector_metadata = renderer.render(
                vector_components, view, ax=ax, **kwargs.get("vector_options", {})
            )
            metadata["vector"] = vector_metadata

        # 出力パスの生成
        filepath = self.config.get_output_path(name, timestamp)

        # 描画結果の出力
        self.exporter.export(fig, filepath, metadata)

        return str(filepath)
