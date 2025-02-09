"""圧力場計算のテストスクリプト

このスクリプトは、圧力ポアソン方程式の右辺を計算し、
ポアソン方程式のソルバを適用して圧力場を求めます。
"""

import yaml
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from simulations import SimulationInitializer
from visualization import visualize_simulation_state
from physics.poisson.sor import SORSolver
from physics.poisson.pressure_rhs import PoissonRHSComputer
from logger import SimulationLogger, LogConfig


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="圧力場計算テスト")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="設定ファイルのパス"
    )
    parser.add_argument(
        "--output", type=str, default="pressure_test", help="出力ディレクトリ名"
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="最大反復回数")
    parser.add_argument(
        "--save-interval", type=int, default=100, help="可視化結果の保存間隔"
    )
    return parser.parse_args()


def setup_visualization(config: dict, output_dir: Path):
    """可視化の設定

    Args:
        config: 設定辞書
        output_dir: 出力ディレクトリ

    Returns:
        更新された設定辞書
    """
    viz_config = config.get("visualization", {})
    viz_config["output_dir"] = str(output_dir)
    viz_config["format"] = "png"
    viz_config["dpi"] = 300
    return config


def save_pressure_field(pressure, iteration: int, output_dir: Path):
    """圧力場を可視化して保存

    Args:
        pressure: 圧力場
        iteration: 現在の反復回数
        output_dir: 出力ディレクトリ
    """
    plt.figure(figsize=(10, 8))

    # 圧力場の中央断面をプロット
    slice_idx = pressure.shape[2] // 2
    plt.imshow(
        pressure[:, :, slice_idx].T, origin="lower", cmap="RdBu_r", aspect="equal"
    )
    plt.colorbar(label="Pressure")
    plt.title(f"Pressure Field (Iteration {iteration})")
    plt.xlabel("X")
    plt.ylabel("Y")

    # 保存
    plt.savefig(output_dir / f"pressure_{iteration:04d}.png")
    plt.close()


def main():
    # コマンドライン引数の解析
    args = parse_args()

    # 出力ディレクトリの作成
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 設定ファイルの読み込み
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ロガーの設定
    logger = SimulationLogger(
        "PressureTest", LogConfig(level="info", log_dir=output_dir)
    )

    # 可視化設定の更新
    config = setup_visualization(config, output_dir)

    try:
        # 初期状態の生成
        logger.info("初期状態を生成中...")
        initializer = SimulationInitializer(config, logger)
        state = initializer.create_initial_state()

        # ポアソンソルバの設定
        solver_config = config.get("solver", {}).get("pressure_solver", {})
        poisson_solver = SORSolver(
            omega=solver_config.get("omega", 1.5),
            tolerance=solver_config.get("tolerance", 1e-6),
            max_iterations=args.max_iter,
            use_redblack=True,
            auto_tune=True,
            boundary_conditions=None,  # または適切な境界条件を設定
            logger=logger,
        )

        # 右辺計算機の初期化
        rhs_computer = PoissonRHSComputer()

        # 初期状態の可視化
        logger.info("初期状態を可視化...")
        visualize_simulation_state(state, config)
        save_pressure_field(state.pressure.data, 0, output_dir)

        # メインループ
        logger.info("圧力場の計算を開始...")
        initial_residual = None

        # 右辺の計算
        rhs = rhs_computer.compute(state.velocity, state.levelset, state.properties)

        for iteration in range(1, args.max_iter + 1):
            # ポアソン方程式を解く
            new_pressure = poisson_solver.solve(
                initial_solution=state.pressure.data, rhs=rhs.data, dx=state.velocity.dx
            )

            # 圧力場の更新
            state.pressure.data = new_pressure

            # 残差の計算と記録
            residual = poisson_solver.residual_history[-1]
            if initial_residual is None:
                initial_residual = residual
            relative_residual = residual / initial_residual

            # 進捗の出力
            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}: "
                    f"Residual = {residual:.3e}, "
                    f"Relative Residual = {relative_residual:.3e}"
                )

            # 可視化結果の保存
            if iteration % args.save_interval == 0:
                save_pressure_field(state.pressure.data, iteration, output_dir)
                visualize_simulation_state(state, config, timestamp=float(iteration))

            # 収束判定
            if relative_residual < solver_config.get("tolerance", 1e-6):
                logger.info(f"収束達成（{iteration}回の反復後）")
                break

        # 最終状態の可視化
        logger.info("最終状態を可視化...")
        save_pressure_field(state.pressure.data, args.max_iter, output_dir)
        visualize_simulation_state(state, config, timestamp=float(args.max_iter))

        # 収束の履歴をプロット
        plt.figure(figsize=(10, 6))
        plt.semilogy(poisson_solver.residual_history)
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.title("Convergence History")
        plt.savefig(output_dir / "convergence_history.png")
        plt.close()

        return 0

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        raise


if __name__ == "__main__":
    main()
