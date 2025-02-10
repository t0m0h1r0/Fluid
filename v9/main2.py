"""二相流シミュレーションの実行スクリプト"""

import sys
import argparse
from pathlib import Path

from logger import SimulationLogger, LogConfig
from simulations import TwoPhaseFlowSimulator, SimulationConfig
from visualization import visualize_simulation_state
from typing import Optional


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Level Set法による二相流シミュレーション"
    )
    parser.add_argument("--config", type=str, required=True, help="設定ファイルのパス")
    parser.add_argument("--checkpoint", type=str, help="チェックポイントファイルのパス")
    parser.add_argument("--debug", action="store_true", help="デバッグモードを有効化")
    return parser.parse_args()


def setup_logging(config: SimulationConfig, debug: bool) -> SimulationLogger:
    """ロギングを設定"""
    log_level = "debug" if debug else "info"
    log_dir = Path(config.output.directory) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SimulationLogger("TwoPhaseFlow", LogConfig(level=log_level, log_dir=log_dir))


def initialize_simulation(
    config: SimulationConfig,
    logger: SimulationLogger,
    checkpoint: Optional[Path] = None,
) -> TwoPhaseFlowSimulator:
    """シミュレーションを初期化"""
    # 出力ディレクトリの作成
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        logger.info(f"チェックポイントから再開: {checkpoint}")
        sim = TwoPhaseFlowSimulator(config, logger)
        sim.load_checkpoint()
        return sim
    else:
        logger.info("新規シミュレーションを開始")
        sim = TwoPhaseFlowSimulator(config, logger)
        sim.initialize()
        return sim


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    args = parse_args()

    # 設定ファイルの読み込み
    config = SimulationConfig.from_yaml(args.config)

    # ロガーの設定
    logger = setup_logging(config, args.debug)

    # チェックポイントファイルのパス
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    try:
        # シミュレーションの初期化
        sim = initialize_simulation(config, logger, checkpoint)

        # 初期状態の可視化と保存
        state, _ = sim.get_state()
        visualize_simulation_state(state, config, timestamp=0.0)
        sim.save_checkpoint()

        # シミュレーションパラメータ
        save_interval = config.time.save_interval
        max_time = config.time.max_time
        next_save_time = save_interval

        logger.info(
            f"シミュレーション開始:\n"
            f"  最大時間: {max_time} [s]\n"
            f"  保存間隔: {save_interval} [s]"
        )

        while next_save_time <= max_time:
            try:
                # 時間発展の実行
                state, step_info = sim.step_forward()

                # 結果の保存
                if step_info["time"] >= next_save_time:
                    visualize_simulation_state(
                        state, config, timestamp=step_info["time"]
                    )
                    sim.save_checkpoint()
                    next_save_time += save_interval

                # 進捗の出力
                logger.info(
                    f"Time: {step_info['time']:.3f}/{max_time:.1f} "
                    f"(dt={step_info['dt']:.3e}), "
                    f"Diagnostics: {step_info['diagnostics']}"
                )

            except Exception as e:
                logger.error(f"シミュレーションステップ中にエラー: {e}")
                break

        logger.info("シミュレーション正常終了")
        return 0

    except Exception as e:
        logger.error(f"実行中にエラーが発生: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
