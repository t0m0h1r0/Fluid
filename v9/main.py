#!/usr/bin/env python3
"""二相流体シミュレーションのメインプログラム

Level Set法を用いた二相流体シミュレーションを実行します。
"""

import sys
import argparse
from pathlib import Path
import yaml
import traceback

from logger import SimulationLogger, LogConfig
from simulations import SimulationInitializer, SimulationRunner, SimulationMonitor
from visualization import StateVisualizer


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
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (IOError, yaml.YAMLError) as e:
        print(f"設定ファイルの読み込み中にエラーが発生: {e}", file=sys.stderr)
        sys.exit(1)


def setup_logging(config: dict, debug: bool = False) -> SimulationLogger:
    """ロギングを設定

    Args:
        config: 設定辞書
        debug: デバッグモードフラグ

    Returns:
        設定されたロガー
    """
    # ログレベルの設定
    log_level = "debug" if debug else config.get("logging", {}).get("level", "info")

    # 出力ディレクトリの設定
    log_dir = Path(config.get("visualization", {}).get("output_dir", "results/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # ログ設定の作成
    log_config = LogConfig(
        level=log_level,
        log_dir=log_dir,
    )

    return SimulationLogger("TwoPhaseFlow", log_config)


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    args = parse_args()

    try:
        # 設定の読み込み
        config = load_config(args.config)

        # ロギングの設定
        logger = setup_logging(config, args.debug)
        logger.start_section("main")

        try:
            # 出力ディレクトリの準備
            output_dir = Path(
                config.get("visualization", {}).get(
                    "output_dir", "results/visualization"
                )
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # シミュレーション監視の初期化
            with SimulationMonitor(config, logger) as monitor:
                # チェックポイントからの再開または新規シミュレーション
                if args.checkpoint:
                    # チェックポイントから再開
                    logger.info(f"チェックポイントから再開: {args.checkpoint}")
                    runner, state = SimulationRunner.from_checkpoint(
                        Path(args.checkpoint), config, logger
                    )
                else:
                    # 新規シミュレーション
                    logger.info("新規シミュレーションを開始")

                    # 初期状態の生成
                    initializer = SimulationInitializer(config, logger)
                    state = initializer.create_initial_state()

                    # シミュレーションランナーの初期化
                    runner = SimulationRunner(config, logger, monitor)

                # 初期状態の可視化
                visualizer = StateVisualizer(config, logger)
                visualizer.visualize(state)

                # シミュレーション実行
                final_state = runner.run(state, output_dir)

                # 最終状態の可視化
                visualizer.visualize(final_state)

                # 結果の解析とレポート生成
                monitor.plot_history(output_dir)
                monitor.generate_report(output_dir)

                # シミュレーションの概要を取得
                summary = monitor.get_summary()
                logger.info("シミュレーションが正常に完了しました")
                logger.info(f"概要: {summary}")

                return 0

        except Exception as e:
            logger.log_error_with_context(
                "シミュレーション中にエラーが発生",
                e,
                {"traceback": traceback.format_exc()},
            )
            return 1

    except Exception as e:
        print(f"シミュレーションの初期化に失敗: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
