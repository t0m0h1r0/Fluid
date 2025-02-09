import sys
import argparse
import yaml
from pathlib import Path
from logger import SimulationLogger, LogConfig
from simulations import SimulationManager, SimulationInitializer
from visualization import visualize_simulation_state


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="二相流シミュレーション")
    parser.add_argument("--config", type=str, required=True, help="設定ファイルのパス")
    parser.add_argument("--checkpoint", type=str, help="チェックポイントファイルのパス")
    parser.add_argument("--debug", action="store_true", help="デバッグモードを有効化")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict, debug: bool) -> SimulationLogger:
    """ロギングを設定"""
    log_level = "debug" if debug else config.get("debug", {}).get("level", "info")
    log_dir = Path(config.get("visualization", {}).get("output_dir", "results"))
    log_dir.mkdir(parents=True, exist_ok=True)
    return SimulationLogger("TwoPhaseFlow", LogConfig(level=log_level, log_dir=log_dir))


def initialize_simulation(config: dict, logger: SimulationLogger, checkpoint: Path = None):
    """シミュレーションを初期化"""
    output_dir = Path(config["visualization"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        logger.info(f"チェックポイントから再開: {checkpoint}")
        # チェックポイントから状態を復元する処理
        initializer = SimulationInitializer(config, logger)
        runner, state = initializer.load_checkpoint(checkpoint)
    else:
        logger.info("新規シミュレーションを開始")
        initializer = SimulationInitializer(config, logger)
        state = initializer.create_initial_state()
        manager = SimulationManager(config, logger)
        runner = manager.runner
        runner.initialize(state)

    # 初期状態の可視化
    visualize_simulation_state(state, config, timestamp=0.0)

    return runner, state


def run_simulation(runner, initial_state, config, logger):
    """シミュレーションを実行"""
    # シミュレーションパラメータの取得
    save_interval = config["numerical"].get("save_interval", 0.1)
    max_time = config["numerical"].get("max_time", 1.0)
    output_dir = Path(config["visualization"]["output_dir"])

    current_time = 0.0
    next_save_time = save_interval

    logger.info(f"シミュレーションを開始: max_time = {max_time}, save_interval = {save_interval}")

    while current_time < max_time:
        try:
            # シミュレーションステップを進める
            state, step_info = runner.step_forward()
            current_time = step_info.get("time", current_time)

            # 結果の可視化
            visualize_simulation_state(state, config, timestamp=current_time)

            # チェックポイントの保存
            if current_time >= next_save_time:
                checkpoint_path = output_dir / f"checkpoint_{current_time:.3f}.npz"
                runner.save_checkpoint(checkpoint_path)
                next_save_time += save_interval

            # ログに進捗を出力
            logger.info(
                f"ステップ {step_info.get('step', 0)}: "
                f"t={current_time:.3f}, dt={step_info.get('dt', 0):.3e}"
            )

        except Exception as e:
            logger.error(f"シミュレーションステップ中にエラー: {e}")
            break

    # シミュレーション終了処理
    runner.finalize(output_dir)
    logger.info("シミュレーション正常終了")


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    args = parse_args()

    # 設定ファイルの読み込み
    config = load_config(args.config)

    # ロガーの設定
    logger = setup_logging(config, args.debug)

    # チェックポイントファイルのパス
    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    try:
        # シミュレーションの初期化
        runner, initial_state = initialize_simulation(config, logger, checkpoint)

        # シミュレーションの実行
        run_simulation(runner, initial_state, config, logger)

        return 0

    except Exception as e:
        logger.error(f"シミュレーション中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())