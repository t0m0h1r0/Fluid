#!/usr/bin/env python3
"""二相流体シミュレーションのメインプログラム

Level Set法を用いた二相流体シミュレーションを実行し、
初期状態の可視化を柔軟に行います。
"""

import sys
import argparse
from pathlib import Path
import yaml
import traceback
import numpy as np

from logger import SimulationLogger, LogConfig
from simulations import SimulationInitializer, SimulationRunner, SimulationMonitor
from visualization import Visualizer2D


class InitialStateVisualizer:
    """初期状態可視化クラス"""

    def __init__(self, config: dict, logger):
        """初期化

        Args:
            config: 設定辞書
            logger: ロガー
        """
        self.config = config.get("visualization", {}).get("initial_state", {})
        self.logger = logger
        self.output_dir = Path(config["visualization"]["output_dir"]) / "initial_state"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_slice_params(self, data_shape):
        """スライスパラメータを取得

        Args:
            data_shape: データの形状

        Returns:
            スライスのパラメータ
        """
        slice_config = self.config.get("slice", {})
        slice_axis = slice_config.get("axis", 2)
        slice_index = data_shape[slice_axis] // 2
        return slice_axis, slice_index

    def _visualize_2d(self, state):
        """2Dスライスの可視化

        Args:
            state: シミュレーションの初期状態
        """
        try:
            # 2D可視化システムを作成
            visualizer = Visualizer2D()
            fields_config = self.config.get("fields", {})

            # 速度場の可視化
            if fields_config.get("velocity", {}).get("enabled", True) and hasattr(
                state, "velocity"
            ):
                velocity_components = state.velocity.components

                # ベクトルプロット
                if "vector" in fields_config["velocity"].get("plot_types", []):
                    visualizer.visualize_vector(
                        velocity_components, "initial_velocity_vector", timestamp=0.0
                    )

                # 速度大きさプロット
                if "magnitude" in fields_config["velocity"].get("plot_types", []):
                    magnitude = np.sqrt(sum(c.data**2 for c in velocity_components))
                    visualizer.visualize_scalar(
                        magnitude, "initial_velocity_magnitude", timestamp=0.0
                    )

            # 圧力場の可視化
            if fields_config.get("pressure", {}).get("enabled", True) and hasattr(
                state, "pressure"
            ):
                pressure_data = state.pressure.data
                slice_axis, slice_index = self._get_slice_params(pressure_data.shape)

                # スカラープロット
                if "scalar" in fields_config["pressure"].get("plot_types", []):
                    visualizer.visualize_scalar(
                        pressure_data,
                        "initial_pressure",
                        timestamp=0.0,
                        slice_axis=slice_axis,
                        slice_index=slice_index,
                    )

                # 等高線プロット
                if "contour" in fields_config["pressure"].get("plot_types", []):
                    visualizer.visualize_scalar(
                        pressure_data,
                        "initial_pressure_contour",
                        timestamp=0.0,
                        slice_axis=slice_axis,
                        slice_index=slice_index,
                        contour=True,
                    )

            # レベルセット場の可視化
            if fields_config.get("levelset", {}).get("enabled", True) and hasattr(
                state, "levelset"
            ):
                levelset_data = state.levelset.data
                slice_axis, slice_index = self._get_slice_params(levelset_data.shape)

                # 界面プロット
                if "interface" in fields_config["levelset"].get("plot_types", []):
                    visualizer.visualize_interface(
                        levelset_data,
                        timestamp=0.0,
                        slice_axis=slice_axis,
                        slice_index=slice_index,
                    )

                # 等高線プロット
                if "contour" in fields_config["levelset"].get("plot_types", []):
                    visualizer.visualize_scalar(
                        levelset_data,
                        "initial_levelset_contour",
                        timestamp=0.0,
                        slice_axis=slice_axis,
                        slice_index=slice_index,
                        contour=True,
                    )

            # 複合フィールドの可視化
            if (
                fields_config.get("combined", {}).get("enabled", True)
                and hasattr(state, "pressure")
                and hasattr(state, "velocity")
                and hasattr(state, "levelset")
            ):
                visualizer.visualize_combined(
                    {
                        "pressure": state.pressure.data,
                        "velocity_x": state.velocity.components[0].data,
                        "velocity_y": state.velocity.components[1].data,
                        "levelset": state.levelset.data,
                    },
                    timestamp=0.0,
                )

            self.logger.info("2D初期状態の可視化が完了しました")

        except Exception as e:
            self.logger.warning(f"2D初期状態の可視化中にエラーが発生: {e}")
            traceback.print_exc()

    def _visualize_3d(self, state):
        """3D可視化の準備（VTK出力）

        Args:
            state: シミュレーションの初期状態
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
                self.logger.warning("PyEVTK not installed. Skipping 3D visualization.")
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
            vtk_path = self.output_dir / f"initial_state.{output_format}"

            # データの次元を取得
            shape = list(list(vtk_data.values())[0].shape)

            # グリッド座標を生成
            x = np.linspace(0, 1, shape[0])
            y = np.linspace(0, 1, shape[1])
            z = np.linspace(0, 1, shape[2])

            # VTKファイルに出力
            gridToVTK(str(vtk_path), x, y, z, pointData=vtk_data)

            self.logger.info(f"3D初期状態をVTKファイルとして出力: {vtk_path}")

        except Exception as e:
            self.logger.warning(f"3D初期状態の可視化中にエラーが発生: {e}")
            traceback.print_exc()

    def visualize(self, state):
        """初期状態を可視化

        Args:
            state: シミュレーションの初期状態
        """
        # 有効な次元を取得
        dimensions = self.config.get("dimensions", ["2D"])

        # 2D可視化
        if "2D" in dimensions:
            self._visualize_2d(state)

        # 3D可視化
        if "3D" in dimensions:
            self._visualize_3d(state)


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="Two-phase flow simulation using Level Set method"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint file to resume from"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定辞書
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    args = parse_args()

    try:
        # 設定の読み込み
        config = load_config(args.config)

        # ロギングの設定
        log_config = LogConfig(
            level="debug" if args.debug else "info",
            log_dir=Path(config["visualization"]["output_dir"]) / "logs",
        )
        logger = SimulationLogger("TwoPhaseFlow", log_config)
        logger.start_section("main")

        try:
            # 出力ディレクトリの準備
            output_dir = Path(config["visualization"]["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            with SimulationMonitor(config, logger) as monitor:
                if args.checkpoint:
                    # チェックポイントから再開
                    logger.info(f"Resuming from checkpoint: {args.checkpoint}")
                    runner, state = SimulationRunner.from_checkpoint(
                        Path(args.checkpoint), config, logger
                    )
                else:
                    # 新規シミュレーション
                    logger.info("Starting new simulation")
                    initializer = SimulationInitializer(config, logger)
                    state = initializer.create_initial_state()
                    runner = SimulationRunner(config, logger, monitor)

                # 初期状態の可視化
                initial_visualizer = InitialStateVisualizer(config, logger)
                initial_visualizer.visualize(state)

                # シミュレーション実行
                final_state = runner.run(state, output_dir)

                # 結果の解析とレポート生成
                monitor.plot_history(output_dir)
                monitor.generate_report(output_dir)

                summary = monitor.get_summary()
                logger.info("Simulation completed successfully")
                logger.info(f"Summary: {summary}")

                return 0

        except Exception as e:
            logger.log_error_with_context(
                "Error during simulation", e, {"traceback": traceback.format_exc()}
            )
            return 1

    except Exception as e:
        print(f"Failed to initialize simulation: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
