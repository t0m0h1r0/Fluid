from pathlib import Path
import numpy as np

from grid_manager import GridConfig, GridManager
from physics_model import PhysicalParameters, PhysicsModel
from boundary_conditions import BoundaryConditionHandler, BoundaryType
from numerical_solver import NumericalSolver
from time_evolution_solver import TimeEvolutionSolver
from visualization_module import SimulationVisualizer

def run_simulation():
    """シミュレーションの実行"""
    # 出力ディレクトリの設定
    output_dir = Path("simulation_results")
    output_dir.mkdir(exist_ok=True)

    # グリッド設定
    grid_config = GridConfig(
        nx=32, ny=32, nz=64,
        Lx=1.0, Ly=1.0, Lz=2.0
    )
    grid_manager = GridManager(grid_config)

    # 物理パラメータ
    physical_params = PhysicalParameters()

    # 境界条件の設定
    boundary_handler = BoundaryConditionHandler(grid_config.grid_shape)
    boundary_handler.set_boundary_condition('x_min', BoundaryType.NEUMANN)
    boundary_handler.set_boundary_condition('x_max', BoundaryType.NEUMANN)
    boundary_handler.set_boundary_condition('y_min', BoundaryType.NEUMANN)
    boundary_handler.set_boundary_condition('y_max', BoundaryType.NEUMANN)
    boundary_handler.set_boundary_condition('z_min', BoundaryType.NEUMANN)
    boundary_handler.set_boundary_condition('z_max', BoundaryType.NEUMANN)

    # 物理モデルの初期条件生成
    physics_model = PhysicsModel(physical_params)
    initial_conditions = physics_model.generate_initial_conditions(
        grid_manager.x, 
        grid_manager.y, 
        grid_manager.z
    )

    # 数値ソルバーの初期化
    numerical_solver = NumericalSolver(grid_config.grid_shape)

    # 時間発展ソルバーの設定
    time_evolution_solver = TimeEvolutionSolver(
        numerical_solver, 
        boundary_handler, 
        physical_params
    )

    # シミュレーション実行
    time_config = {
        'time_step': 0.001,
        'max_steps': 1000
    }
    simulation_history = time_evolution_solver.simulate(
        initial_conditions, 
        time_config
    )

    # 可視化
    visualizer = SimulationVisualizer(output_dir)
    
    # 密度分布の可視化
    for step, density in enumerate(simulation_history['density'][::10]):
        visualizer.plot_density_distribution(
            density, 
            step * 10, 
            {
                'Lx': grid_config.Lx, 
                'Ly': grid_config.Ly, 
                'Lz': grid_config.Lz
            }
        )
    
    # 速度場の可視化
    for step in range(0, len(simulation_history['velocity_u']), 10):
        visualizer.plot_velocity_field(
            simulation_history['velocity_u'][step],
            simulation_history['velocity_v'][step],
            simulation_history['velocity_w'][step],
            step,
            {
                'Lx': grid_config.Lx, 
                'Ly': grid_config.Ly, 
                'Lz': grid_config.Lz
            }
        )
    
    # データ保存
    visualizer.save_simulation_data(simulation_history)
    
    # 時系列解析
    visualizer.plot_time_series(simulation_history, 'density')
    visualizer.plot_time_series(simulation_history, 'pressure')

if __name__ == "__main__":
    run_simulation()
