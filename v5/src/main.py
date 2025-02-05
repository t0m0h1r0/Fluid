# main.py
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from physics.navier_stokes import NavierStokesSolver
from physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from numerics.compact_scheme import CompactScheme
from numerics.poisson_solver.classic_poisson_solver import (
    ClassicPoissonSolver as MultigridPoissonSolver,
)
from core.boundary import DirectionalBC, PeriodicBC, NeumannBC
from physics.fluid_properties import MultiPhaseProperties, FluidProperties
from data_io.data_writer import DataWriter
from visualization.visualizer import SimulationVisualizer
from core.simulation import SimulationManager
from utils.config import SimulationConfig


class SimulationRunner:
    def __init__(self, config_path: str):
        """シミュレーション実行クラスの初期化

        Args:
            config_path: 設定ファイルのパス
        """
        self.config_path = config_path
        self.output_dir = self._setup_output_directory()
        self.config = SimulationConfig(config_path)

        # 各コンポーネントの初期化
        self.writer = DataWriter(self.output_dir)
        self.visualizer = SimulationVisualizer(self.output_dir)
        self.manager = self._setup_simulation_manager()

        # 実行状態の初期化
        self.current_step = 0
        self.is_running = False

    def _setup_output_directory(self) -> Path:
        """出力ディレクトリの設定"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # 設定ファイルのコピー
        config_backup = output_dir / "simulation_config.yaml"
        config_backup.write_bytes(Path(self.config_path).read_bytes())

        return output_dir

    def _setup_simulation_manager(self) -> SimulationManager:
        """シミュレーションマネージャーの設定"""
        # スキームと境界条件の設定
        scheme = CompactScheme()
        boundary_conditions = DirectionalBC(
            x_bc=PeriodicBC(), y_bc=PeriodicBC(), z_bc=NeumannBC()
        )

        # ポアソンソルバーの初期化
        poisson_solver = MultigridPoissonSolver(
            scheme=scheme,
            boundary_conditions=[
                boundary_conditions.get_condition(i) for i in range(3)
            ],
        )

        # 流体物性の設定
        fluids = {
            phase.name: FluidProperties(phase.name, phase.density, phase.viscosity)
            for phase in self.config.phases
        }
        fluid_properties = MultiPhaseProperties(fluids)

        # ソルバーの初期化
        phase_solver = PhaseFieldSolver(
            scheme=scheme,
            boundary_conditions=boundary_conditions,
            params=PhaseFieldParams(),
        )

        ns_solver = NavierStokesSolver(
            scheme=scheme,
            boundary_conditions=boundary_conditions,
            poisson_solver=poisson_solver,
            fluid_properties=fluid_properties,
        )

        # フェーズ密度の設定
        for phase in self.config.phases:
            phase_solver.set_phase_density(phase.name, phase.density)
            print(f"密度を設定: {phase.name} = {phase.density} kg/m³")

        return SimulationManager(self.config, ns_solver, phase_solver, fluid_properties)

    def run(self, checkpoint_interval: int = 100):
        """シミュレーションの実行"""
        print("シミュレーションを開始します...")
        self.is_running = True

        try:
            # 初期状態の保存と可視化
            self._save_and_visualize(0)

            while self.current_step < self.config.max_steps:
                # 1ステップ進める
                results = self.manager.advance_timestep(adaptive_dt=True)
                self.current_step += 1

                # 進捗表示
                if self.current_step % 10 == 0:
                    self._print_progress(results)

                # チェックポイントの保存
                if self.current_step % checkpoint_interval == 0:
                    self._save_and_visualize(self.current_step)

                # 収束判定
                if self._check_convergence(results):
                    print("\n収束条件を満たしました。シミュレーションを終了します。")
                    break

            # 最終状態の保存
            self._save_and_visualize(self.current_step)

        except KeyboardInterrupt:
            print("\nシミュレーションが中断されました。")
            self._save_and_visualize(self.current_step)

        except Exception as e:
            print(f"\nエラーが発生しました: {str(e)}")
            raise

        finally:
            self.is_running = False
            print("\nシミュレーションが完了しました。")

    def _save_and_visualize(self, step: int):
        """結果の保存と可視化"""
        # データの保存
        self.writer.save_state(
            self.manager.phi, self.manager.velocity, self.manager.pressure, step
        )

        # 可視化
        self.visualizer.create_visualization(
            {
                "phi": self.manager.phi,
                "velocity": self.manager.velocity,
                "pressure": self.manager.pressure,
                "stats": self.manager.stats.__dict__,
            },
            step,
            save=True,
        )

    def _print_progress(self, results: Dict[str, Any]):
        """進捗情報の表示"""
        stats = self.manager.stats
        print(f"\nステップ {self.current_step} (t = {stats.total_time:.3f}s)")
        print(f"平均時間刻み幅: {stats.avg_timestep:.2e}s")
        print(f"最大速度: {stats.max_velocity:.2e}m/s")
        print(f"エネルギー保存: {stats.energy_conservation:.6f}")
        print(f"質量保存: {stats.mass_conservation:.6f}")

    def _check_convergence(self, results: Dict[str, Any]) -> bool:
        """収束判定"""
        stats = self.manager.stats

        # エネルギーと質量の保存性チェック
        energy_error = abs(1.0 - stats.energy_conservation)
        mass_error = abs(1.0 - stats.mass_conservation)

        # 速度の最大値チェック
        velocity_max = stats.max_velocity

        return energy_error < 1e-6 and mass_error < 1e-8 and velocity_max < 1e-6


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="二層流シミュレーション")
    parser.add_argument(
        "--config", default="config/simulation.yaml", help="設定ファイルのパス"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100, help="チェックポイント保存間隔"
    )
    args = parser.parse_args()

    # シミュレーションの実行
    runner = SimulationRunner(args.config)
    runner.run(checkpoint_interval=args.checkpoint_interval)


if __name__ == "__main__":
    main()
