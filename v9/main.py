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
from simulations import SimulationManager


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
    log_level = "debug" if debug else config.get("debug", {}).get("level", "info")

    # 出力ディレクトリの設定
    log_dir = Path(config.get("visualization", {}).get("output_dir", "results"))
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

        # シミュレーション管理クラスの初期化
        manager = SimulationManager(config, logger)

        # シミュレーションの実行
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        exit_code = manager.run_simulation(checkpoint)

        # シミュレーション終了時の処理
        manager.cleanup()

        return exit_code

    except Exception as e:
        print(f"シミュレーションの実行中にエラーが発生: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
