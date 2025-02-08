"""シミュレーション状態の可視化を管理するモジュール

任意のシミュレーション状態を柔軟に可視化するためのクラスを提供します。
"""

import traceback
from pathlib import Path
from typing import Dict, Any, List, Protocol

import numpy as np

from simulations.state import SimulationState


class DataSliceStrategy(Protocol):
    """データスライス戦略のプロトコル"""

    def slice_data(self, data: np.ndarray, axis: int, index: int) -> np.ndarray:
        """データをスライスする"""
        ...


class DefaultDataSliceStrategy:
    """デフォルトのデータスライス戦略"""

    def slice_data(self, data: np.ndarray, axis: int, index: int) -> np.ndarray:
        """データから2Dスライスを抽出

        Args:
            data: 入力データ
            axis: スライスする軸
            index: スライスのインデックス

        Returns:
            2Dスライス
        """
        # 2Dデータの場合はそのまま返す
        if data.ndim == 2:
            return data

        # スライスを取得
        slices = [slice(None)] * data.ndim
        slices[axis] = index

        return data[tuple(slices)]


class FieldVisualizationConfig:
    """フィールド可視化設定"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 可視化設定の辞書
        """
        self.fields = config.get("fields", {})
        self.dimensions = config.get("dimensions", ["2D"])
        self.output_dir = Path(config.get("output_dir", "results/visualization"))
        self.output_dir.mkdir(parents=True, exist_ok=True)


class SliceConfiguration:
    """スライス設定を管理するクラス"""

    @staticmethod
    def get_default_slices(state: SimulationState) -> List[Dict[str, Any]]:
        """デフォルトのスライス面を取得

        Args:
            state: シミュレーション状態

        Returns:
            スライス面のリスト
        """
        # データのシェイプを取得
        shape = state.levelset.data.shape

        return [
            {"axis": 2, "name": "XY", "slice_index": shape[2] // 2},  # Z軸に垂直な面
            {"axis": 1, "name": "XZ", "slice_index": shape[1] // 2},  # Y軸に垂直な面
            {"axis": 0, "name": "YZ", "slice_index": shape[0] // 2},  # X軸に垂直な面
        ]


class FieldVisualizer:
    """特定のフィールドを可視化するクラス"""

    def __init__(
        self,
        visualizer,  # Type hintを削除して循環インポートを回避
        slice_strategy: DataSliceStrategy = DefaultDataSliceStrategy(),
    ):
        """
        Args:
            visualizer: 可視化に使用するVisualizer2Dインスタンス
            slice_strategy: データスライス戦略
        """
        self._visualizer = visualizer
        self._slice_strategy = slice_strategy

    def visualize_velocity(
        self,
        velocity,
        plane: Dict[str, Any],
        timestamp: float,
        plot_types: List[str] = None,
    ):
        """速度場の可視化

        Args:
            velocity: 速度場
            plane: スライス面の情報
            timestamp: 現在の時刻
            plot_types: 可視化するプロットの種類
        """
        plot_types = plot_types or ["vector", "magnitude"]

        # ベクトルプロット
        if "vector" in plot_types:
            components = [
                self._slice_strategy.slice_data(
                    comp.data, plane["axis"], plane["slice_index"]
                )
                for comp in velocity.components[:2]
            ]
            self._visualizer.visualize_vector(
                components,
                f"velocity_vector_{plane['name']}",
                timestamp=timestamp,
                slice_axis=plane["axis"],
                slice_index=plane["slice_index"],
            )

        # 速度大きさプロット
        if "magnitude" in plot_types:
            magnitude = np.sqrt(
                sum(
                    self._slice_strategy.slice_data(
                        comp.data, plane["axis"], plane["slice_index"]
                    )
                    ** 2
                    for comp in velocity.components[:2]
                )
            )
            self._visualizer.visualize_scalar(
                magnitude,
                f"velocity_magnitude_{plane['name']}",
                timestamp=timestamp,
                slice_axis=plane["axis"],
                slice_index=plane["slice_index"],
            )

    def visualize_pressure(
        self,
        pressure: np.ndarray,
        plane: Dict[str, Any],
        timestamp: float,
        plot_types: List[str] = None,
    ):
        """圧力場の可視化

        Args:
            pressure: 圧力場データ
            plane: スライス面の情報
            timestamp: 現在の時刻
            plot_types: 可視化するプロットの種類
        """
        plot_types = plot_types or ["scalar", "contour"]
        sliced_data = self._slice_strategy.slice_data(
            pressure, plane["axis"], plane["slice_index"]
        )

        # スカラープロット
        if "scalar" in plot_types:
            self._visualizer.visualize_scalar(
                sliced_data,
                f"pressure_{plane['name']}",
                timestamp=timestamp,
                slice_axis=plane["axis"],
                slice_index=plane["slice_index"],
            )

        # 等高線プロット
        if "contour" in plot_types:
            self._visualizer.visualize_scalar(
                sliced_data,
                f"pressure_contour_{plane['name']}",
                timestamp=timestamp,
                slice_axis=plane["axis"],
                slice_index=plane["slice_index"],
                contour=True,
            )

    def visualize_levelset(
        self,
        levelset: np.ndarray,
        plane: Dict[str, Any],
        timestamp: float,
        plot_types: List[str] = None,
    ):
        """Level Set場の可視化

        Args:
            levelset: Level Set場データ
            plane: スライス面の情報
            timestamp: 現在の時刻
            plot_types: 可視化するプロットの種類
        """
        plot_types = plot_types or ["interface", "contour"]
        sliced_data = self._slice_strategy.slice_data(
            levelset, plane["axis"], plane["slice_index"]
        )

        # 界面プロット
        if "interface" in plot_types:
            self._visualizer.visualize_interface(
                sliced_data,
                timestamp=timestamp,
                slice_axis=plane["axis"],
                slice_index=plane["slice_index"],
                name=f"levelset_interface_{plane['name']}",
            )

        # 等高線プロット
        if "contour" in plot_types:
            self._visualizer.visualize_scalar(
                sliced_data,
                f"levelset_contour_{plane['name']}",
                timestamp=timestamp,
                slice_axis=plane["axis"],
                slice_index=plane["slice_index"],
                contour=True,
                symmetric=True,  # Level Set関数は対称な色範囲
            )


class StateVisualizer:
    """シミュレーション状態の可視化を管理するクラス"""

    def __init__(self, config: Dict[str, Any], logger=None):
        """初期化

        Args:
            config: 設定辞書
            logger: ロガーオブジェクト
        """
        # 設定の初期化
        self.config = FieldVisualizationConfig(config.get("visualization", {}))
        self.logger = logger

        # 可視化クラスの初期化（遅延インポート）
        from visualization import Visualizer2D

        visualizer = Visualizer2D()
        self._field_visualizer = FieldVisualizer(visualizer)

    def visualize(self, state: SimulationState, timestamp: float = 0.0):
        """状態を可視化

        Args:
            state: シミュレーションの状態
            timestamp: 可視化する時刻（デフォルトは0.0）
        """
        try:
            # 必須フィールドの検証
            self._validate_state(state)

            # 2D可視化
            if "2D" in self.config.dimensions:
                # デフォルトのスライス面を取得
                planes = SliceConfiguration.get_default_slices(state)

                # 各面を可視化
                for plane in planes:
                    # フィールドごとの可視化
                    self._visualize_fields(state, plane, timestamp)

        except Exception as e:
            if self.logger:
                self.logger.error(f"状態の可視化中にエラーが発生: {e}")
                traceback.print_exc()

    def _validate_state(self, state: SimulationState):
        """シミュレーション状態の妥当性を検証

        Args:
            state: 検証するシミュレーション状態

        Raises:
            ValueError: 必要なフィールドが不足している場合
        """
        required_fields = ["velocity", "pressure", "levelset"]
        for field in required_fields:
            if not hasattr(state, field):
                raise ValueError(
                    f"シミュレーション状態に必須のフィールド '{field}' が不足しています"
                )

    def _visualize_fields(
        self, state: SimulationState, plane: Dict[str, Any], timestamp: float
    ):
        """各フィールドを可視化

        Args:
            state: シミュレーション状態
            plane: 可視化する面の情報
            timestamp: 現在の時刻
        """
        # 各フィールドの可視化設定を取得
        fields_config = self.config.fields

        # 速度場の可視化
        velocity_config = fields_config.get("velocity", {})
        if velocity_config.get("enabled", True):
            self._field_visualizer.visualize_velocity(
                state.velocity, plane, timestamp, velocity_config.get("plot_types")
            )

        # 圧力場の可視化
        pressure_config = fields_config.get("pressure", {})
        if pressure_config.get("enabled", True):
            self._field_visualizer.visualize_pressure(
                state.pressure.data, plane, timestamp, pressure_config.get("plot_types")
            )

        # Level Set場の可視化
        levelset_config = fields_config.get("levelset", {})
        if levelset_config.get("enabled", True):
            self._field_visualizer.visualize_levelset(
                state.levelset.data, plane, timestamp, levelset_config.get("plot_types")
            )
