import argparse
from pathlib import Path
from typing import Optional

from config.base import Config
from config.validator import ConfigValidator
from core.simulation import Simulation
from core.material.properties import MaterialManager
from core.solver.navier_stokes.solver import NavierStokesSolver
from core.solver.time_integrator.runge_kutta import RK4
from core.solver.poisson.multigrid import MultiGrid
from physics.phase_field import PhaseField, PhaseFieldParameters
from data_io.hdf5_io import HDF5IO
from data_io.csv_io import CSVIO
from data_io.visualizer import Visualizer

class SimulationRunner:
    def __init__(self, config_path: Path):
        self.config = Config.from_yaml(config_path)
        self._validate_config()
        
        self.io = self._create_io()
        self.visualizer = Visualizer(self.config.output.output_dir)
        self.simulation = self._create_simulation()

    def _validate_config(self):
        errors = ConfigValidator.validate(self.config)
        if errors:
            raise ValueError("\n".join(errors))

    def _create_io(self):
        if self.config.output.save_format == 'hdf5':
            return HDF5IO(self.config.output.output_dir)
        else:
            return CSVIO(self.config.output.output_dir)

    def _create_simulation(self):
        # 物性値管理
        material_manager = MaterialManager()
        for name, fluid in self.config.fluids.items():
            material_manager.add_fluid(name, fluid)

        # Phase-Field
        phase_field = PhaseField(
            parameters=PhaseFieldParameters(
                epsilon=self.config.phase.epsilon,
                mobility=self.config.phase.mobility,
                surface_tension=self.config.phase.surface_tension
            )
        )

        # ソルバー
        poisson_solver = MultiGrid(
            levels=3,
            v_cycles=3,
            tolerance=self.config.numerical.tolerance,
            max_iterations=self.config.numerical.max_iterations
        )

        ns_solver = NavierStokesSolver(
            poisson_solver=poisson_solver
        )

        # 時間積分
        time_integrator = RK4()

        return Simulation(
            material_manager=material_manager,
            phase_field=phase_field,
            ns_solver=ns_solver,
            time_integrator=time_integrator,
            config=self.config
        )

    def run(self):
        print("シミュレーションを開始します...")
        try:
            # 初期状態の保存と可視化
            self._save_results(0)
            print("初期状態を保存しました")
            
            timestep = 0
            time = 0.0
            next_save = self.config.output.save_interval  # 次の保存時刻

            print("時間発展計算を開始します...")
            while time < self.config.numerical.max_time:
                print(f"\rTime: {time:.3f} s", end="")

                # シミュレーションステップ
                self.simulation.step(self.config.numerical.dt)
                time += self.config.numerical.dt
                timestep += 1

                # 結果の保存
                if time >= next_save:
                    self._save_results(timestep)
                    next_save += self.config.output.save_interval
                    print(f"\n時刻 {time:.3f} s の結果を保存しました")

            # 最終状態の保存
            self._save_results(timestep)
            print("\n最終状態を保存しました")
            print("\nシミュレーションが完了しました")

        except KeyboardInterrupt:
            print("\nシミュレーションが中断されました")
            # 中断時の状態を保存
            self._save_results(timestep)
            print("中断時の状態を保存しました")
            
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            raise

    def _save_results(self, timestep: int):
        # フィールドデータの保存
        field_data = self.simulation.get_field_data()
        for name, field in field_data.items():
            filepath = self.io.save_field(field, timestep)
            print(f"  - {name}フィールドを保存: {filepath}")

        # 可視化
        if self.config.output.visualization:
            # 相場の可視化
            phase_plot = self.visualizer.create_slice_plot(
                field_data['phase'], timestep,
                cmap='RdYlBu'
            )
            print(f"  - 相場の可視化を保存: {phase_plot}")

            # 速度場の可視化
            velocity_plot = self.visualizer.create_vector_plot(
                field_data['velocity'], timestep,
                cmap='coolwarm', scale=50
            )
            print(f"  - 速度場の可視化を保存: {velocity_plot}")

            # 圧力場の可視化
            pressure_plot = self.visualizer.create_slice_plot(
                field_data['pressure'], timestep,
                cmap='RdBu'
            )
            print(f"  - 圧力場の可視化を保存: {pressure_plot}")

            # 密度場の可視化
            density_plot = self.visualizer.create_slice_plot(
                field_data['density'], timestep,
                cmap='viridis'
            )
            print(f"  - 密度場の可視化を保存: {density_plot}")

def main():
    parser = argparse.ArgumentParser(description='二相流体シミュレーション')
    parser.add_argument('--config', type=Path, default='config/default_config.yaml',
                       help='設定ファイルのパス')
    args = parser.parse_args()

    runner = SimulationRunner(args.config)
    runner.run()

if __name__ == "__main__":
    main()