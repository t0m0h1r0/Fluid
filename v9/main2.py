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
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SimulationLogger("TwoPhaseFlow", LogConfig(level=log_level, log_dir=log_dir))


def initialize_simulation(
    config: SimulationConfig,
    logger: SimulationLogger,
    checkpoint: Optional[Path] = None,
) -> TwoPhaseFlowSimulator:
    """シミュレーションを初期化"""
    # 出力ディレクトリの作成
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        logger.info(f"チェックポイントから再開: {checkpoint}")
        sim = TwoPhaseFlowSimulator(config, logger)
        sim.initialize(state=sim.load_checkpoint(str(checkpoint)))
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

        # 初期チェックポイントを保存（オプション）
        output_dir = Path(config.output_dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        initial_checkpoint = output_dir / "initial_checkpoint.npz"
        sim.save_checkpoint(str(initial_checkpoint))

        # シミュレーションパラメータ
        save_interval = config.time.save_interval
        max_time = config.time.max_time
        next_save_time = save_interval

        # 最初の時間刻み幅を計算
        current_dt = sim._time_solver.compute_timestep(state=state)

        logger.info(
            f"シミュレーション開始:\n"
            f"  最大時間: {max_time} [s]\n"
            f"  保存間隔: {save_interval} [s]\n"
            f"  初期時間刻み幅: {current_dt:.3e} [s]"
        )

        while next_save_time <= max_time:
            try:
                # 時間発展の実行（時間刻み幅を明示的に渡す）
                result = sim._time_solver.step_forward(dt=current_dt, state=state)
                state = result["state"]
                step_info = result.get("diagnostics", {})

                # 結果の保存
                if step_info.get("time", 0.0) >= next_save_time:
                    visualize_simulation_state(
                        state, config, timestamp=step_info["time"]
                    )

                    # チェックポイントを保存
                    output_dir = Path(config.output_dir) / "checkpoints"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_filename = f"checkpoint_{step_info['time']:.4f}.npz"
                    checkpoint_path = output_dir / checkpoint_filename
                    sim.save_checkpoint(str(checkpoint_path))

                    next_save_time += save_interval

                # 次のステップの時間刻み幅を計算
                current_dt = sim._time_solver.compute_timestep(state=state)

                # 進捗の出力
                logger.info(
                    f"Time: {step_info.get('time', 0.0):.3f}/{max_time:.1f} "
                    f"(dt={current_dt:.3e}), "
                    f"Diagnostics: {step_info}"
                )

            except Exception as e:
                logger.error(f"シミュレーションステップ中にエラー: {e}")
                import traceback

                traceback.print_exc()
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
