"""
二相流シミュレーションのメインスクリプト

このスクリプトは、Level Set法を用いた二相流体シミュレーションの実行を管理します。
主な機能:
- コマンドライン引数の解析
- シミュレーション設定の読み込み
- シミュレーションの初期化と実行
- 結果の可視化とチェックポイントの保存
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

from simulations import TwoPhaseFlowSimulator, SimulationConfig, SimulationInitializer
from visualization import visualize_simulation_state


class SimulationManager:
    """
    シミュレーションの実行を管理するクラス

    コマンドライン引数の解析、設定の読み込み、
    シミュレーションの初期化と実行を担当します。
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str] = None,
        debug: bool = False,
    ):
        """
        シミュレーションマネージャを初期化

        Args:
            config_path: シミュレーション設定ファイルのパス
            checkpoint_path: チェックポイントファイルのパス（オプション）
            debug: デバッグモードフラグ
        """
        # 設定の読み込み
        self.config = SimulationConfig.from_yaml(config_path)

        # 出力とチェックポイントディレクトリの準備
        self.output_dir = Path(self.config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 初期化子の作成
        self.initializer = SimulationInitializer(self.config)

        # シミュレータの準備
        self.simulator = self._prepare_simulator(checkpoint_path)

        # デバッグ設定
        self.debug = debug

    def _prepare_simulator(
        self, checkpoint_path: Optional[str] = None
    ) -> TwoPhaseFlowSimulator:
        """
        シミュレータを準備

        Args:
            checkpoint_path: チェックポイントファイルのパス

        Returns:
            初期化されたTwoPhaseFlowSimulator
        """
        # シミュレータの作成
        simulator = TwoPhaseFlowSimulator(config=self.config)

        # チェックポイントからの再開または新規初期化
        if checkpoint_path:
            state = simulator.load_checkpoint(checkpoint_path)
        else:
            state = self.initializer.create_initial_state()

        simulator.initialize(state)
        return simulator

    def run(self) -> int:
        """
        シミュレーションを実行

        Returns:
            終了ステータス (0: 正常終了, 1: エラー)
        """
        try:
            # シミュレーションパラメータ
            save_interval = self.config.numerical.save_interval
            max_time = self.config.numerical.max_time

            # 初期状態の取得と可視化
            current_state, initial_diagnostics = self.simulator.get_state()
            visualize_simulation_state(current_state, self.config, timestamp=0.0)

            # 初期チェックポイントの保存
            initial_checkpoint = self.checkpoint_dir / "initial_checkpoint.npz"
            self.simulator.save_checkpoint(str(initial_checkpoint))

            # シミュレーションのメインループ
            next_save_time = save_interval

            print(
                f"シミュレーション開始:\n"
                f"  最大時間: {max_time} [s]\n"
                f"  保存間隔: {save_interval} [s]"
            )

            while current_state.time < max_time:
                try:
                    # 時間発展の実行
                    new_state, step_info = self.simulator.step_forward()
                    current_state = new_state

                    # 結果の保存と可視化
                    if current_state.time >= next_save_time:
                        # 可視化
                        visualize_simulation_state(
                            current_state, self.config, timestamp=current_state.time
                        )

                        # チェックポイントの保存
                        checkpoint_filename = f"checkpoint_{current_state.time:.4f}.npz"
                        checkpoint_path = self.checkpoint_dir / checkpoint_filename
                        self.simulator.save_checkpoint(str(checkpoint_path))

                        next_save_time += save_interval

                    # 進捗の出力
                    print(
                        f"Time: {current_state.time:.3f}/{max_time:.1f}, "
                        f"Diagnostics: {step_info}"
                    )

                except Exception as step_error:
                    print(
                        f"シミュレーションステップ中にエラー: {step_error}",
                        file=sys.stderr,
                    )
                    import traceback

                    traceback.print_exc()
                    return 1

            print("シミュレーション正常終了")
            return 0

        except Exception as e:
            print(f"シミュレーション実行中にエラー: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1


def parse_args() -> Tuple[str, Optional[str], bool]:
    """
    コマンドライン引数をパース

    Returns:
        設定ファイルパス、チェックポイントファイルパス、デバッグフラグ
    """
    parser = argparse.ArgumentParser(
        description="Level Set法による二相流シミュレーション"
    )
    parser.add_argument("--config", type=str, required=True, help="設定ファイルのパス")
    parser.add_argument("--checkpoint", type=str, help="チェックポイントファイルのパス")
    parser.add_argument("--debug", action="store_true", help="デバッグモードを有効化")

    args = parser.parse_args()
    return args.config, args.checkpoint, args.debug


def main() -> int:
    """
    メイン関数

    Returns:
        終了ステータス (0: 正常終了, 1: エラー)
    """
    # コマンドライン引数の解析
    config_path, checkpoint_path, debug_mode = parse_args()

    try:
        # シミュレーションマネージャの作成と実行
        manager = SimulationManager(config_path, checkpoint_path, debug_mode)
        return manager.run()

    except Exception as e:
        print(f"実行中にエラーが発生: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
