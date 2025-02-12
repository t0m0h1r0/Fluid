"""
二相流シミュレーションのメインスクリプト

このスクリプトは、Level Set法を用いた二相流体シミュレーションの実行を管理します。
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

from simulations import TwoPhaseFlowSimulator, SimulationConfig, SimulationInitializer
from visualization import visualize_simulation_state


class SimulationManager:
    """シミュレーションの実行を管理するクラス"""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        debug: bool = False,
    ):
        """シミュレーションマネージャを初期化"""
        # ロガーの設定
        self.logger = self._setup_logger(debug)

        # 設定の読み込み
        self.logger.info("設定ファイルを読み込み中: %s", config_path)
        self.config = SimulationConfig.from_yaml(config_path)

        # 出力ディレクトリの準備
        self.output_dir = Path(self.config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 診断情報の保存先
        self.diagnostics_dir = self.output_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        # 初期化子の作成
        self.initializer = SimulationInitializer(self.config)

        # シミュレータの準備
        self.simulator = self._prepare_simulator(checkpoint_path)

        # デバッグ設定
        self.debug = debug

    def _setup_logger(self, debug: bool) -> logging.Logger:
        """ロガーを設定"""
        logger = logging.getLogger("SimulationManager")
        logger.setLevel(logging.DEBUG if debug else logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _prepare_simulator(
        self, checkpoint_path: Optional[str] = None
    ) -> TwoPhaseFlowSimulator:
        """シミュレータを準備"""
        simulator = TwoPhaseFlowSimulator(config=self.config)

        if checkpoint_path:
            self.logger.info(
                "チェックポイントから状態を読み込み中: %s", checkpoint_path
            )
            state = simulator.load_checkpoint(checkpoint_path)
        else:
            self.logger.info("新規シミュレーション状態を作成中")
            state = self.initializer.create_initial_state()

        simulator.initialize(state)
        return simulator

    def _save_diagnostics(self, step_info: Dict, current_time: float):
        """診断情報を保存"""
        diagnostics_file = self.diagnostics_dir / f"diagnostics_{current_time:.6f}.json"
        with open(diagnostics_file, "w") as f:
            json.dump(step_info, f, indent=2)

    def run(self) -> int:
        """シミュレーションを実行"""
        try:
            # シミュレーションパラメータ
            save_interval = self.config.numerical.save_interval
            max_time = self.config.numerical.max_time

            # 初期状態の取得と可視化
            current_state, initial_diagnostics = self.simulator.get_state()
            self.logger.info("初期状態を可視化中")
            visualize_simulation_state(current_state, self.config, timestamp=0.0)

            # 初期チェックポイントの保存
            initial_checkpoint = self.checkpoint_dir / "initial_checkpoint.npz"
            self.simulator.save_checkpoint(str(initial_checkpoint))
            self._save_diagnostics(initial_diagnostics, 0.0)

            # シミュレーションのメインループ
            next_save_time = save_interval
            self.logger.info("シミュレーション開始")

            while current_state.time < max_time:
                try:
                    # 時間発展の実行
                    new_state, step_info = self.simulator.step_forward()
                    current_state = new_state

                    # 結果の保存と可視化
                    if current_state.time >= next_save_time:
                        self.logger.info(
                            "時刻 %.6f での状態を保存中", current_state.time
                        )

                        # 可視化
                        visualize_simulation_state(
                            current_state, self.config, timestamp=current_state.time
                        )

                        # チェックポイントの保存
                        checkpoint_filename = f"checkpoint_{current_state.time:.6f}.npz"
                        checkpoint_path = self.checkpoint_dir / checkpoint_filename
                        self.simulator.save_checkpoint(str(checkpoint_path))

                        # 診断情報の保存
                        self._save_diagnostics(step_info, current_state.time)

                        next_save_time += save_interval
                        return 0

                except Exception as step_error:
                    self.logger.error(
                        "シミュレーションステップ中にエラー: %s",
                        step_error,
                        exc_info=True,
                    )
                    return 1

            self.logger.info("シミュレーション正常終了")
            return 0

        except Exception as e:
            self.logger.error("シミュレーション実行中にエラー: %s", e, exc_info=True)
            return 1


def parse_args() -> Tuple[str, Optional[str], bool]:
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Level Set法による二相流シミュレーション"
    )
    parser.add_argument("--config", type=str, required=True, help="設定ファイルのパス")
    parser.add_argument("--checkpoint", type=str, help="チェックポイントファイルのパス")
    parser.add_argument("--debug", action="store_true", help="デバッグモードを有効化")

    args = parser.parse_args()
    return args.config, args.checkpoint, args.debug


def main() -> int:
    """メイン関数"""
    try:
        # コマンドライン引数の解析
        config_path, checkpoint_path, debug_mode = parse_args()

        # シミュレーションマネージャの作成と実行
        manager = SimulationManager(config_path, checkpoint_path, debug_mode)
        return manager.run()

    except Exception as e:
        print(f"実行中にエラーが発生: {e}", file=sys.stderr)
        if debug_mode:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
