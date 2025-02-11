"""二相流シミュレーションの実行スクリプト"""

import sys
import argparse
from pathlib import Path

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


def initialize_simulation(
    config: SimulationConfig,
    checkpoint: Optional[Path] = None,
) -> TwoPhaseFlowSimulator:
    """シミュレーションを初期化

    Args:
        config: シミュレーション設定
        checkpoint: チェックポイントファイルのパス（オプション）

    Returns:
        初期化されたシミュレーター
    """
    # 出力ディレクトリの作成
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # チェックポイントからの再開かどうかで初期化を分岐
    if checkpoint:
        print(f"チェックポイントから再開: {checkpoint}")
        sim = TwoPhaseFlowSimulator(config)
        sim.initialize(state=sim.load_checkpoint(str(checkpoint)))
    else:
        print("新規シミュレーションを開始")
        sim = TwoPhaseFlowSimulator(config)
        sim.initialize()

    return sim


def main():
    """メイン関数"""
    try:
        # コマンドライン引数の解析
        args = parse_args()

        # 設定ファイルの読み込み
        config = SimulationConfig.from_yaml(args.config)

        # チェックポイントファイルのパス
        checkpoint = Path(args.checkpoint) if args.checkpoint else None

        # シミュレーションの初期化
        sim = initialize_simulation(config, checkpoint)

        # 初期状態の可視化と保存
        state, _ = sim.get_state()
        visualize_simulation_state(state, config, timestamp=0.0)

        # 初期チェックポイントを保存
        output_dir = Path(config.output.output_dir) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        initial_checkpoint = output_dir / "initial_checkpoint.npz"
        sim.save_checkpoint(str(initial_checkpoint))

        # シミュレーションパラメータ
        save_interval = config.numerical.save_interval
        max_time = config.numerical.max_time
        next_save_time = save_interval

        # 最初の時間刻み幅を計算
        current_dt = sim._time_solver.compute_timestep(state=state)

        print(
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
                    checkpoint_filename = f"checkpoint_{step_info['time']:.4f}.npz"
                    checkpoint_path = output_dir / checkpoint_filename
                    sim.save_checkpoint(str(checkpoint_path))

                    next_save_time += save_interval

                # 次のステップの時間刻み幅を計算
                current_dt = sim._time_solver.compute_timestep(state=state)

                # 進捗の出力
                print(
                    f"Time: {step_info.get('time', 0.0):.3f}/{max_time:.1f} "
                    f"(dt={current_dt:.3e}), "
                    f"Diagnostics: {step_info}"
                )
                return

            except Exception as e:
                print(f"シミュレーションステップ中にエラー: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                break

        print("シミュレーション正常終了")
        return 0

    except Exception as e:
        print(f"実行中にエラーが発生: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
