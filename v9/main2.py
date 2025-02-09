"""二相流シミュレーションの実行スクリプト

このスクリプトは、Level Set法を用いた二相流シミュレーションを実行します。
設定ファイルの読み込み、シミュレーションの実行、結果の出力を管理します。
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from logger import SimulationLogger, LogConfig
from twoflow import (
    TwoPhaseFlowSimulation,
    SimulationConfig
)
from visualization import visualize_simulation_state


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Level Set法による二相流シミュレーション"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="設定ファイルのパス"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="チェックポイントファイルのパス"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモードを有効化"
    )
    return parser.parse_args()


def setup_logging(config: SimulationConfig, debug: bool) -> SimulationLogger:
    """ロギングを設定

    Args:
        config: シミュレーション設定
        debug: デバッグモードフラグ

    Returns:
        設定されたロガー
    """
    log_level = "debug" if debug else "info"
    log_dir = Path(config.output.directory) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SimulationLogger(
        "TwoPhaseFlow",
        LogConfig(level=log_level, log_dir=log_dir)
    )


def initialize_simulation(
    config: SimulationConfig,
    logger: SimulationLogger,
    checkpoint: Optional[Path] = None
) -> TwoPhaseFlowSimulation:
    """シミュレーションを初期化

    Args:
        config: シミュレーション設定
        logger: ロガー
        checkpoint: チェックポイントファイルのパス（オプション）

    Returns:
        初期化されたシミュレーション
    """
    # 出力ディレクトリの作成
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        logger.info(f"チェックポイントから再開: {checkpoint}")
        return TwoPhaseFlowSimulation.load_checkpoint(
            config,
            checkpoint,
            logger
        )
    else:
        logger.info("新規シミュレーションを開始")
        sim = TwoPhaseFlowSimulation(config, logger)
        sim.initialize()
        return sim


def run_simulation(
    sim: TwoPhaseFlowSimulation,
    config: SimulationConfig,
    logger: SimulationLogger
) -> int:
    """シミュレーションを実行

    Args:
        sim: シミュレーション
        config: シミュレーション設定
        logger: ロガー

    Returns:
        終了コード（0: 成功, 1: 失敗）
    """
    try:
        # シミュレーションパラメータ
        max_time = config.time.max_time
        save_interval = config.time.save_interval
        output_dir = Path(config.output.directory)

        # 初期状態の可視化と保存
        state, _ = sim.get_state()
        visualize_simulation_state(state, config, timestamp=0.0)
        sim.save_checkpoint(output_dir / "checkpoint_initial.npz")

        current_time = 0.0
        next_save_time = save_interval

        logger.info(
            f"シミュレーション開始:\n"
            f"  最大時間: {max_time} [s]\n"
            f"  保存間隔: {save_interval} [s]"
        )

        while current_time < max_time:
            try:
                # 時間発展の実行
                state, step_info = sim.step_forward()
                current_time = step_info["time"]

                # 結果の保存
                if current_time >= next_save_time:
                    visualize_simulation_state(
                        state,
                        config,
                        timestamp=current_time
                    )
                    checkpoint_path = (
                        output_dir /
                        f"checkpoint_{current_time:.3f}.npz"
                    )
                    sim.save_checkpoint(checkpoint_path)
                    next_save_time += save_interval

                # 進捗の出力
                dt = step_info["dt"]
                ns_info = step_info["navier_stokes"]
                ls_info = step_info["level_set"]
                logger.info(
                    f"Time: {current_time:.3f}/{max_time:.1f} "
                    f"(dt={dt:.3e})\n"
                    f"  最大速度: {ns_info['max_velocity']:.3e}\n"
                    f"  圧力範囲: [{ns_info['pressure']['min']:.3e}, "
                    f"{ns_info['pressure']['max']:.3e}]\n"
                    f"  体積誤差: {ls_info['volume_error']:.3e}"
                )

            except Exception as e:
                logger.error(f"シミュレーションステップ中にエラー: {e}")
                break

        # 最終状態の保存
        checkpoint_path = output_dir / "checkpoint_final.npz"
        sim.save_checkpoint(checkpoint_path)
        logger.info("シミュレーション正常終了")
        return 0

    except Exception as e:
        logger.error(f"シミュレーション実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


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
        # シミュレーションの初期化と実行
        sim = initialize_simulation(config, logger, checkpoint)
        return run_simulation(sim, config, logger)

    except Exception as e:
        logger.error(f"実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())