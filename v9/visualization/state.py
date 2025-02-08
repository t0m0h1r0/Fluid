"""シミュレーション状態の可視化モジュール

このモジュールは、シミュレーション状態の可視化を管理するクラスを提供します。
"""

from typing import Dict, Any, Optional
import logging
import numpy as np

from visualization.core.base import VisualizationConfig, ViewConfig
from visualization.visualizer import Visualizer


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
        # 可視化設定の初期化
        vis_config = VisualizationConfig.from_dict(config.get("visualization", {}))
        self.visualizer = Visualizer(vis_config)
        self.logger = logger

    def visualize(
        self,
        state,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
    ) -> None:
        """状態を可視化

        Args:
            state: シミュレーション状態
            timestamp: 現在の時刻
            view: 視点設定
        """
        try:
            # デフォルトのビュー設定
            if view is None:
                view = ViewConfig()

            # 速度場の可視化
            if hasattr(state, "velocity"):
                try:
                    # 速度成分を3次元配列として正しく取得
                    components = []
                    for comp in state.velocity.components:
                        data = comp.data
                        # 必要に応じて3次元に拡張
                        if data.ndim == 1:
                            nx = int(np.sqrt(len(data)))
                            data = data.reshape(nx, nx, 1)
                        elif data.ndim == 2:
                            data = data[..., np.newaxis]
                        components.append(data)

                    # 各スライスでの可視化
                    for slice_type, pos in [("XY", 0.5), ("XZ", 0.5), ("YZ", 0.5)]:
                        self.visualizer.visualize_vector(
                            [comp for comp in components],
                            f"velocity_{slice_type}_{timestamp:.3f}",
                            timestamp=timestamp,
                            view=ViewConfig(
                                slice_axes=[slice_type.lower()], slice_positions=[pos]
                            ),
                            color="k",  # 黒色を指定
                            alpha=0.7,
                            scale=1.0,
                            density=20,
                            magnitude_colors=False,  # 単色表示に設定
                        )

                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"速度場の{slice_type}スライス可視化中にエラー: {str(e)}"
                        )

            # 圧力場の可視化
            if hasattr(state, "pressure"):
                try:
                    pressure_data = state.pressure.data
                    # 必要に応じて3次元に拡張
                    if pressure_data.ndim == 1:
                        nx = int(np.sqrt(len(pressure_data)))
                        pressure_data = pressure_data.reshape(nx, nx, 1)
                    elif pressure_data.ndim == 2:
                        pressure_data = pressure_data[..., np.newaxis]

                    for slice_type, pos in [("XY", 0.5), ("XZ", 0.5), ("YZ", 0.5)]:
                        self.visualizer.visualize_scalar(
                            pressure_data,
                            f"pressure_{slice_type}_{timestamp:.3f}",
                            timestamp=timestamp,
                            view=ViewConfig(
                                slice_axes=[slice_type.lower()], slice_positions=[pos]
                            ),
                            colorbar_label="Pressure",
                        )

                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"圧力場の{slice_type}スライス可視化中にエラー: {str(e)}"
                        )

            # Level Set場の可視化
            if hasattr(state, "levelset"):
                try:
                    levelset_data = state.levelset.data
                    # 必要に応じて3次元に拡張
                    if levelset_data.ndim == 1:
                        nx = int(np.sqrt(len(levelset_data)))
                        levelset_data = levelset_data.reshape(nx, nx, 1)
                    elif levelset_data.ndim == 2:
                        levelset_data = levelset_data[..., np.newaxis]

                    for slice_type, pos in [("XY", 0.5), ("XZ", 0.5), ("YZ", 0.5)]:
                        self.visualizer.visualize_scalar(
                            levelset_data,
                            f"levelset_{slice_type}_{timestamp:.3f}",
                            timestamp=timestamp,
                            view=ViewConfig(
                                slice_axes=[slice_type.lower()], slice_positions=[pos]
                            ),
                            colorbar_label="Level Set",
                            contour=True,
                            levels=[0],
                            colors=["black"],
                        )

                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Level Set場の{slice_type}スライス可視化中にエラー: {str(e)}"
                        )

            # 組み合わせ可視化
            try:
                # 3次元データの準備
                velocity_components = None
                if hasattr(state, "velocity"):
                    velocity_components = [
                        self._ensure_3d(comp.data) for comp in state.velocity.components
                    ]

                pressure_data = None
                if hasattr(state, "pressure"):
                    pressure_data = self._ensure_3d(state.pressure.data)

                self.visualizer.visualize_combined(
                    scalar_data=pressure_data,
                    vector_components=velocity_components,
                    name=f"combined_{timestamp:.3f}",
                    timestamp=timestamp,
                    view=view,
                    scalar_options={"colorbar_label": "Pressure"},
                    vector_options={
                        "color": "k",
                        "alpha": 0.7,
                        "scale": 1.0,
                        "density": 20,
                        "magnitude_colors": False,
                    },
                )

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"組み合わせ可視化中にエラー: {str(e)}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"可視化中にエラーが発生: {str(e)}")
            raise

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
