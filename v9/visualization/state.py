"""シミュレーション状態の可視化モジュール

このモジュールは、シミュレーション状態の可視化を管理するクラスを提供します。
"""

from typing import Dict, Any, Optional
import logging
import numpy as np
import copy

from simulations.state import SimulationState
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
        # 可視化設定を取得（安全にアクセス）
        fields_config = self.visualizer.config.fields or {}

        # 3Dデータのスライス位置を定義
        slice_positions = [0.5, 0.5, 0.5]  # デフォルトは中央スライス
        slice_names = ["XY", "XZ", "YZ"]
        slice_axes = [2, 1, 0]  # それぞれのスライス面に対応する軸

        # 各スライス面での可視化
        for axis, (slice_pos, slice_name) in enumerate(
            zip(slice_positions, slice_names)
        ):
            slice_view = copy.deepcopy(view) if view is not None else ViewConfig()
            slice_view.slice_position = slice_pos

            # 速度場の可視化
            if hasattr(state, "velocity") and fields_config.get("velocity", {}).get(
                "enabled", True
            ):
                try:
                    components = [comp.data for comp in state.velocity.components]

                    # 3Dデータから2Dスライスを取得
                    slice_components = [
                        np.take(comp, int(slice_pos * comp.shape[axis]), axis=axis)
                        for comp in components[
                            :2
                        ]  # 2Dスライスのため最初の2成分のみを使用
                    ]

                    for plot_type in fields_config.get("velocity", {}).get(
                        "plot_types", []
                    ):
                        if plot_type == "vector":
                            # デフォルトの色は黒に設定
                            vector_plot_params = {
                                "color": "k",
                                "magnitude_colors": False,
                            }
                            self.visualizer.visualize_vector(
                                slice_components,
                                f"velocity_vector_{slice_name}",
                                timestamp=timestamp,
                                view=slice_view,
                                **vector_plot_params,
                            )
                        elif plot_type == "magnitude":
                            magnitude = np.sqrt(
                                sum(comp**2 for comp in slice_components)
                            )
                            self.visualizer.visualize_scalar(
                                magnitude,
                                f"velocity_magnitude_{slice_name}",
                                timestamp=timestamp,
                                view=slice_view,
                            )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"速度場の{slice_name}スライス可視化中にエラー: {str(e)}"
                        )

            # 圧力場の可視化
            if hasattr(state, "pressure") and fields_config.get("pressure", {}).get(
                "enabled", True
            ):
                try:
                    pressure_slice = np.take(
                        state.pressure.data,
                        int(slice_pos * state.pressure.data.shape[axis]),
                        axis=axis,
                    )

                    for plot_type in fields_config.get("pressure", {}).get(
                        "plot_types", []
                    ):
                        if plot_type == "scalar":
                            self.visualizer.visualize_scalar(
                                pressure_slice,
                                f"pressure_{slice_name}",
                                timestamp=timestamp,
                                view=slice_view,
                            )
                        elif plot_type == "contour":
                            self.visualizer.visualize_scalar(
                                pressure_slice,
                                f"pressure_contour_{slice_name}",
                                timestamp=timestamp,
                                view=slice_view,
                                contour=True,
                            )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"圧力場の{slice_name}スライス可視化中にエラー: {str(e)}"
                        )

            # Level Set場の可視化
            if hasattr(state, "levelset") and fields_config.get("levelset", {}).get(
                "enabled", True
            ):
                try:
                    levelset_slice = np.take(
                        state.levelset.data,
                        int(slice_pos * state.levelset.data.shape[axis]),
                        axis=axis,
                    )

                    for plot_type in fields_config.get("levelset", {}).get(
                        "plot_types", []
                    ):
                        if plot_type == "interface":
                            self.visualizer.visualize_scalar(
                                levelset_slice,
                                f"levelset_interface_{slice_name}",
                                timestamp=timestamp,
                                view=slice_view,
                                contour=True,
                                levels=[0],
                                colors=["black"],
                            )
                        elif plot_type == "contour":
                            self.visualizer.visualize_scalar(
                                levelset_slice,
                                f"levelset_contour_{slice_name}",
                                timestamp=timestamp,
                                view=slice_view,
                                contour=True,
                                symmetric=True,
                            )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Level Set場の{slice_name}スライス可視化中にエラー: {str(e)}"
                        )

        # 必要に応じて、全体の組み合わせ可視化も維持
        try:
            self.visualizer.visualize_combined(
                scalar_data=state.pressure.data if hasattr(state, "pressure") else None,
                vector_components=[comp.data for comp in state.velocity.components]
                if hasattr(state, "velocity")
                else None,
                name="combined",
                timestamp=timestamp,
                view=view,
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"組み合わせ可視化中にエラー: {str(e)}")

    except Exception as e:
        if self.logger:
            self.logger.error(f"可視化中にエラーが発生: {str(e)}")
        raise
