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
from simulations import SimulationManager, SimulationInitializer, SimulationRunner
from visualization import visualize_simulation_state
import logging


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


def should_output_visualization(
    current_time: float,
    current_step: int,
    last_output_time: float,
    last_output_step: int,
    viz_config: dict,
) -> bool:
    """可視化出力が必要かどうかを判定

    Args:
        current_time: 現在の時刻
        current_step: 現在のステップ数
        last_output_time: 前回の出力時刻
        last_output_step: 前回の出力ステップ
        viz_config: 可視化設定

    Returns:
        出力が必要な場合True
    """
    output_control = viz_config.get("output_control", {})
    time_interval = output_control.get("time_interval", 0.0)
    step_interval = output_control.get("every_steps", 0)

    should_output = False

    # 時間間隔での判定
    if time_interval > 0:
        if current_time >= last_output_time + time_interval:
            should_output = True

    # ステップ間隔での判定
    if step_interval > 0:
        if current_step >= last_output_step + step_interval:
            should_output = True

    return should_output


def run_simulation(
    config: dict, logger: SimulationLogger, checkpoint: Path = None
) -> int:
    """シミュレーションを実行

    Args:
        config: 設定辞書
        logger: ロガー
        checkpoint: チェックポイントファイル（オプション）

    Returns:
        終了コード
    """
    try:
        # 出力ディレクトリの設定
        output_dir = Path(config["visualization"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # 可視化の設定
        viz_config = config.get("visualization", {})
        save_interval = config.get("numerical", {}).get("save_interval", 0.1)
        max_time = config.get("numerical", {}).get("max_time", 1.0)

        # 初期状態の設定
        if checkpoint:
            # チェックポイントから再開
            logger.info(f"チェックポイントから再開: {checkpoint}")
            runner, state = SimulationRunner.from_checkpoint(checkpoint, config, logger)
        else:
            # 新規シミュレーション
            logger.info("新規シミュレーションを開始")
            initializer = SimulationInitializer(config, logger)
            manager = SimulationManager(config, logger)
            state = initializer.create_initial_state()
            runner = manager.runner
            runner.initialize(state)

            # 初期状態の可視化（設定で有効な場合）
            if viz_config.get("output_control", {}).get("initial_state", True):
                logger.info("初期状態を可視化")
                visualize_simulation_state(state, config, timestamp=0.0)

        # 次の保存タイミング
        next_save_time = save_interval

        # 可視化の出力管理用の変数
        last_output_time = 0.0
        last_output_step = 0

        # メインループ
        while True:
            # 現在の状態を取得
            status = runner.get_status()
            current_time = status["current_time"]
            current_step = status["current_step"]

            # 終了判定
            if current_time >= max_time:
                break

            # 1ステップ進める
            state, step_info = runner.step_forward()

            # データの保存
            if current_time >= next_save_time:
                checkpoint_path = output_dir / f"checkpoint_{current_time:.3f}.npz"
                runner.save_checkpoint(checkpoint_path)
                next_save_time += save_interval

            # 可視化の出力判定
            if should_output_visualization(
                current_time,
                current_step,
                last_output_time,
                last_output_step,
                viz_config,
            ):
                visualize_simulation_state(state, config, timestamp=current_time)
                last_output_time = current_time
                last_output_step = current_step

            # ログ出力
            if logger.level <= logging.INFO:
                logger.info(
                    f"Step {step_info['step']}: t={current_time:.3f}, "
                    f"dt={step_info['dt']:.3e}"
                )

        # シミュレーション終了時の可視化
        visualize_simulation_state(state, config, timestamp=current_time)

        # 終了処理
        runner.finalize(output_dir)
        logger.info("シミュレーション正常終了")

        return 0

    except Exception as e:
        logger.error(f"シミュレーション実行中にエラーが発生: {e}")
        logger.error(traceback.format_exc())
        return 1


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    args = parse_args()

    try:
        # 設定の読み込み
        config = load_config(args.config)

        # ロギングの設定
        logger = setup_logging(config, args.debug)

        # シミュレーションの実行
        checkpoint = Path(args.checkpoint) if args.checkpoint else None
        exit_code = run_simulation(config, logger, checkpoint)

        return exit_code

    except Exception as e:
        print(f"シミュレーションの実行中にエラーが発生: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
