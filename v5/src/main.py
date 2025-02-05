import numpy as np
from core.simulation import SimulationConfig, SimulationManager
from core.scheme import DifferenceScheme
from core.boundary import DirectionalBC, PeriodicBC, NeumannBC
from numerics.compact_scheme import CompactScheme
from numerics.poisson_solver import PoissonSolver
from physics.fluid_properties import FluidProperties, MultiPhaseProperties
from physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from physics.navier_stokes import NavierStokesSolver
from data_io.visualizer import Visualizer
from data_io.checkpoint import CheckpointHandler

def create_simulation():
    # 設定
    config = SimulationConfig(
        shape=(32, 32, 64),
        dt=0.001,
        save_interval=0.1,
        max_time=1.0,
        sphere_center=(0.5, 0.5, 0.4),
        sphere_radius=0.2,
        top_phase_level=1.5
    )
    
    # スキームと境界条件
    scheme = CompactScheme(alpha=0.25)
    boundary_conditions = DirectionalBC(
        x_bc=PeriodicBC(),
        y_bc=PeriodicBC(),
        z_bc=NeumannBC()
    )
    
    # ポアソンソルバー
    poisson_solver = PoissonSolver(
        scheme=scheme,
        boundary_conditions=[
            boundary_conditions.get_condition(i) 
            for i in range(3)
        ]
    )
    
    # 流体物性
    fluids = {
        'water': FluidProperties(
            density=1000.0,
            viscosity=1.0e-3,
            name='water'
        ),
        'nitrogen': FluidProperties(
            density=1.225,
            viscosity=1.79e-5,
            name='nitrogen'
        )
    }
    fluid_properties = MultiPhaseProperties(fluids)
    
    # ソルバー
    phase_solver = PhaseFieldSolver(
        scheme=scheme,
        boundary_conditions=boundary_conditions,
        params=PhaseFieldParams()
    )
    
    ns_solver = NavierStokesSolver(
        scheme=scheme,
        boundary_conditions=boundary_conditions,
        poisson_solver=poisson_solver,
        fluid_properties=fluid_properties
    )
    
    return SimulationManager(
        config=config,
        navier_stokes=ns_solver,
        phase_field=phase_solver,
        fluid_properties=fluid_properties
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Two-phase flow simulation')
    parser.add_argument('--resume', type=str, help='Checkpoint file to resume from')
    args = parser.parse_args()
    
    # シミュレーション作成
    sim = create_simulation()
    visualizer = Visualizer()
    checkpoint = CheckpointHandler('output')
    
    # チェックポイントからの再開
    if args.resume:
        sim.set_state(checkpoint.load(args.resume))
    
    while sim.time < sim.config.max_time:
        # 時間発展
        fields = sim.advance_timestep()
        
        # 進捗表示と検証
        if sim.step % 100 == 0:
            is_valid, issues = sim.validate_state()
            print(f"t = {sim.time:.3f}, step = {sim.step}")
            if not is_valid:
                print("Warning: Validation issues detected:")
                for issue in issues:
                    print(f"  - {issue}")
        
        # 保存
        if sim.should_save():
            print(f"Saving state at t = {sim.time:.3f}")
            checkpoint.save(sim.get_state(), sim.step)
            visualizer.save_plots(fields, sim.time, sim.step)

if __name__ == "__main__":
    main()