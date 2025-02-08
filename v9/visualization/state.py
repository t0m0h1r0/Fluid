"""シミュレーション状態の可視化を管理するモジュール

任意のシミュレーション状態を柔軟に可視化するためのクラスを提供します。
"""

import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

# 循環インポートを回避するために、直接インポートではなく遅延インポート
from simulations.state import SimulationState


class StateVisualizer:
    """シミュレーション状態の可視化を管理するクラス"""

    def __init__(
        self, config: Dict[str, Any], logger=None, output_dir: Optional[Path] = None
    ):
        """初期化

        Args:
            config: 設定辞書
            logger: ロガーオブジェクト
            output_dir: 出力ディレクトリ（オプション）
        """
        # 可視化設定を取得
        self.config = config.get("visualization", {})
        self.initial_state_config = self.config.get("initial_state", {})
        self.logger = logger

        # 出力ディレクトリの設定
        if output_dir is None:
            base_output_dir = self.config.get("output_dir", "results/visualization")
            output_dir = Path(base_output_dir) / "states"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _get_slice_planes(self, state: SimulationState):
        """可視化する面を取得

        Args:
            state: シミュレーションの状態

        Returns:
            可視化する面のリスト
        """
        slice_config = self.initial_state_config.get("slice", {})

        # デフォルトの面を定義
        planes = [
            {"axis": 2, "name": "XY"},  # Z軸に垂直な面
            {"axis": 1, "name": "XZ"},  # Y軸に垂直な面
            {"axis": 0, "name": "YZ"},  # X軸に垂直な面
        ]

        # カスタム面の設定がある場合は上書き
        if slice_config.get("custom_planes"):
            planes = slice_config["custom_planes"]

        # 各面のスライスインデックスを追加
        for plane in planes:
            plane["slice_index"] = state.levelset.data.shape[plane["axis"]] // 2

        return planes

    def _visualize_field(
        self,
        state: SimulationState,
        plane: Dict[str, Any],
        field_name: str,
        timestamp: float,
    ):
        """特定のフィールドを可視化

        Args:
            state: シミュレーション状態
            plane: 可視化する面の情報
            field_name: 可視化するフィールド名
            timestamp: 現在の時刻
        """
        # 遅延インポート
        from visualization import Visualizer2D

        if not hasattr(state, field_name):
            return

        fields_config = self.initial_state_config.get("fields", {})
        field_config = fields_config.get(field_name, {})

        if not field_config.get("enabled", True):
            return

        field_obj = getattr(state, field_name)
        visualizer = Visualizer2D()

        # フィールドの種類に応じて可視化メソッドを選択
        if field_name == "velocity":
            components = field_obj.components
            plot_types = field_config.get("plot_types", [])

            # ベクトルプロット
            if "vector" in plot_types:
                visualizer.visualize_vector(
                    components,
                    f"velocity_vector_{plane['name']}",
                    timestamp=timestamp,
                    slice_axis=plane["axis"],
                    slice_index=plane["slice_index"],
                )

            # 速度大きさプロット
            if "magnitude" in plot_types:
                magnitude = np.sqrt(sum(c.data**2 for c in components))
                visualizer.visualize_scalar(
                    magnitude,
                    f"velocity_magnitude_{plane['name']}",
                    timestamp=timestamp,
                    slice_axis=plane["axis"],
                    slice_index=plane["slice_index"],
                )

        elif field_name == "pressure":
            plot_types = field_config.get("plot_types", [])
            data = field_obj.data

            # スカラープロット
            if "scalar" in plot_types:
                visualizer.visualize_scalar(
                    data,
                    f"pressure_{plane['name']}",
                    timestamp=timestamp,
                    slice_axis=plane["axis"],
                    slice_index=plane["slice_index"],
                )

            # 等高線プロット
            if "contour" in plot_types:
                visualizer.visualize_scalar(
                    data,
                    f"pressure_contour_{plane['name']}",
                    timestamp=timestamp,
                    slice_axis=plane["axis"],
                    slice_index=plane["slice_index"],
                    contour=True,
                )

        elif field_name == "levelset":
            plot_types = field_config.get("plot_types", [])
            data = field_obj.data

            # 界面プロット
            if "interface" in plot_types:
                visualizer.visualize_interface(
                    data,
                    timestamp=timestamp,
                    slice_axis=plane["axis"],
                    slice_index=plane["slice_index"],
                )

            # 等高線プロット
            if "contour" in plot_types:
                visualizer.visualize_scalar(
                    data,
                    f"levelset_contour_{plane['name']}",
                    timestamp=timestamp,
                    slice_axis=plane["axis"],
                    slice_index=plane["slice_index"],
                    contour=True,
                )

    def _visualize_3d(self, state: SimulationState, timestamp: float):
        """3D可視化の準備（VTK出力）

        Args:
            state: シミュレーションの状態
            timestamp: 現在の時刻
        """
        try:
            # 3D可視化の設定を取得
            vtk_config = self.config.get("3d", {})
            if not vtk_config.get("enabled", True):
                return

            # 必要なライブラリをインポート
            try:
                from pyevtk.hl import gridToVTK
            except ImportError:
                if self.logger:
                    self.logger.warning(
                        "PyEVTKがインストールされていません。3D可視化をスキップします。"
                    )
                return

            # 出力フォーマットを取得
            output_format = vtk_config.get("format", "vti")

            # 可視化するフィールドを取得
            fields_to_visualize = vtk_config.get(
                "fields", ["velocity", "pressure", "levelset"]
            )

            # VTKファイルの出力
            vtk_data = {}

            # 各フィールドを追加
            for field_name in fields_to_visualize:
                if hasattr(state, field_name):
                    field_obj = getattr(state, field_name)

                    # VectorFieldの特別な処理
                    if field_name == "velocity":
                        vtk_data.update(
                            {
                                f"velocity_{comp}": comp.data
                                for comp in field_obj.components
                            }
                        )
                    else:
                        vtk_data[field_name] = field_obj.data

            # VTKファイルのパス
            vtk_filename = f"state_t{timestamp:.6f}.{output_format}"
            vtk_path = self.output_dir / vtk_filename

            # データの次元を取得
            shape = list(list(vtk_data.values())[0].shape)

            # グリッド座標を生成
            x = np.linspace(0, 1, shape[0])
            y = np.linspace(0, 1, shape[1])
            z = np.linspace(0, 1, shape[2])

            # VTKファイルに出力
            gridToVTK(str(vtk_path), x, y, z, pointData=vtk_data)

            if self.logger:
                self.logger.info(f"3D状態をVTKファイルとして出力: {vtk_path}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"3D状態の可視化中にエラーが発生: {e}")
                traceback.print_exc()

    def visualize(self, state: SimulationState, timestamp: float = 0.0):
        """状態を可視化

        Args:
            state: シミュレーションの状態
            timestamp: 可視化する時刻（デフォルトは0.0）
        """
        try:
            # 状態の妥当性を検証
            self._validate_state(state)

            # 有効な次元を取得
            dimensions = self.initial_state_config.get("dimensions", ["2D"])

            # 2D可視化
            if "2D" in dimensions:
                # 可視化する面を取得
                planes = self._get_slice_planes(state)

                # 各面を可視化
                for plane in planes:
                    # 個別のフィールドを可視化
                    for field_name in ["velocity", "pressure", "levelset"]:
                        self._visualize_field(state, plane, field_name, timestamp)

                    # 複合フィールドの可視化
                    if (
                        self.initial_state_config.get("fields", {})
                        .get("combined", {})
                        .get("enabled", True)
                    ):
                        # 遅延インポート
                        from visualization import Visualizer2D

                        Visualizer2D().visualize_combined(
                            {
                                "pressure": state.pressure.data,
                                "velocity_x": state.velocity.components[0].data,
                                "velocity_y": state.velocity.components[1].data,
                                "levelset": state.levelset.data,
                            },
                            timestamp=timestamp,
                            slice_axis=plane["axis"],
                            slice_index=plane["slice_index"],
                            name=f"combined_{plane['name']}",
                        )

            # 3D可視化
            if "3D" in dimensions:
                self._visualize_3d(state, timestamp)

            if self.logger:
                self.logger.info(f"状態の可視化が完了しました (t = {timestamp:.3f})")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"状態の可視化中にエラーが発生: {e}")
                traceback.print_exc()
