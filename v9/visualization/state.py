"""シミュレーション状態の可視化を管理するモジュール

このモジュールは、シミュレーション状態の包括的な可視化を提供します。
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

from simulations.state import SimulationState
from .core.base import VisualizationConfig, ViewConfig
from .visualizer import Visualizer


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
        state: SimulationState,
        timestamp: float = 0.0,
        view: Optional[ViewConfig] = None,
    ) -> None:
        """状態を可視化

        Args:
            state: シミュレーション状態
            timestamp: 現在の時刻
            view: 3D表示用の視点設定
        """
        try:
            # 各フィールドの可視化設定を取得
            fields_config = self.visualizer.config.fields

            # 速度場の可視化
            if hasattr(state, "velocity") and fields_config.get("velocity", {}).get(
                "enabled", True
            ):
                try:
                    components = [comp.data for comp in state.velocity.components]
                    for plot_type in fields_config["velocity"].get("plot_types", []):
                        if plot_type == "vector":
                            self.visualizer.visualize_vector(
                                components,
                                "velocity_vector",
                                timestamp=timestamp,
                                view=view,
                            )
                        elif plot_type == "magnitude":
                            magnitude = np.sqrt(sum(comp**2 for comp in components))
                            self.visualizer.visualize_scalar(
                                magnitude,
                                "velocity_magnitude",
                                timestamp=timestamp,
                                view=view,
                            )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"速度場の可視化中にエラー: {str(e)}")

            # 圧力場の可視化
            if hasattr(state, "pressure") and fields_config.get("pressure", {}).get(
                "enabled", True
            ):
                try:
                    for plot_type in fields_config["pressure"].get("plot_types", []):
                        if plot_type == "scalar":
                            self.visualizer.visualize_scalar(
                                state.pressure.data,
                                "pressure",
                                timestamp=timestamp,
                                view=view,
                            )
                        elif plot_type == "contour":
                            self.visualizer.visualize_scalar(
                                state.pressure.data,
                                "pressure_contour",
                                timestamp=timestamp,
                                view=view,
                                contour=True,
                            )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"圧力場の可視化中にエラー: {str(e)}")

            # Level Set場の可視化
            if hasattr(state, "levelset") and fields_config.get("levelset", {}).get(
                "enabled", True
            ):
                try:
                    for plot_type in fields_config["levelset"].get("plot_types", []):
                        if plot_type == "interface":
                            self.visualizer.visualize_scalar(
                                state.levelset.data,
                                "levelset_interface",
                                timestamp=timestamp,
                                view=view,
                                contour=True,
                                levels=[0],
                                colors=["black"],
                            )
                        elif plot_type == "contour":
                            self.visualizer.visualize_scalar(
                                state.levelset.data,
                                "levelset_contour",
                                timestamp=timestamp,
                                view=view,
                                contour=True,
                                symmetric=True,
                            )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Level Set場の可視化中にエラー: {str(e)}")

            # 組み合わせ可視化
            try:
                self.visualizer.visualize_combined(
                    scalar_data=state.pressure.data
                    if hasattr(state, "pressure")
                    else None,
                    vector_components=[comp.data for comp in state.velocity.components]
                    if hasattr(state, "velocity")
                    else None,
                    timestamp=timestamp,
                    view=view,
                    slice_positions=self.visualizer.config.volume_plot.slice_positions,
                )
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"組み合わせ可視化中にエラー: {str(e)}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"可視化中にエラーが発生: {str(e)}")
            raise
