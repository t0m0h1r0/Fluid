#!/usr/bin/env python3
"""二相流体シミュレーションのメインプログラム

このプログラムは、Level Set法を用いた二相流体シミュレーションを実行します。
コマンドライン引数で設定ファイルを指定し、シミュレーションを実行します。
"""

import sys
import argparse
from pathlib import Path
import yaml
import traceback

from logger import SimulationLogger, LogConfig
from simulations import SimulationInitializer, SimulationRunner, SimulationMonitor


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
