from physics.navier_stokes import NavierStokesSolver
from physics.phase_field import (
    PhaseFieldSolver, PhaseFieldParams,
    Layer, Sphere
)
from numerics.compact_scheme import CompactScheme
from numerics.poisson_solver.classic_poisson_solver import ClassicPoissonSolver as PoissonSolver
from core.boundary import DirectionalBC, PeriodicBC, NeumannBC
from physics.fluid_properties import MultiPhaseProperties, FluidProperties
from data_io.data_writer import DataWriter
from utils.config import SimulationConfig
import numpy as np
import os 

def setup_simulation():
    """シミュレーションの設定とソルバーの初期化を行う
    
    Returns:
        tuple: config, ns_solver, phase_solver, fluid_properties
    """
    config = SimulationConfig('config/simulation.yaml')
    scheme = CompactScheme()
    boundary_conditions = DirectionalBC(
        x_bc=PeriodicBC(), y_bc=PeriodicBC(), z_bc=NeumannBC()
    )
    
    poisson_solver = PoissonSolver(
        scheme=scheme,
        boundary_conditions=[boundary_conditions.get_condition(i) for i in range(3)]
    )
    
    fluids = {phase.name: FluidProperties(phase.name, phase.density, phase.viscosity) 
             for phase in config.phases}
    
    fluid_properties = MultiPhaseProperties(fluids)
    
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
    
    return config, ns_solver, phase_solver, fluid_properties

def initialize_fields(config: SimulationConfig, phase_solver: PhaseFieldSolver):
    """フィールドの初期化を行う
    
    Args:
        config: シミュレーション設定
        phase_solver: PhaseFieldソルバー
        
    Returns:
        tuple: velocity, pressure, density fields
    """
    print("\nInitializing fields...")
    shape = (config.Nx, config.Ny, config.Nz)
    domain_size = (config.Lx, config.Ly, config.Lz)

    # 基本フィールドの初期化
    velocity = [np.zeros(shape) for _ in range(3)]
    pressure = np.zeros(shape)
    
    # 密度場の初期化
    density = phase_solver.initialize_field(shape, domain_size)
    
    # フェーズ密度の登録
    for phase in config.phases:
        phase_solver.set_phase_density(phase.name, phase.density)
        print(f"Registered phase: {phase.name} with density {phase.density}")
    
    # レイヤーの設定
    print("\nSetting up layers...")
    for layer_config in config.layers:
        layer = Layer(
            phase_name=layer_config.phase,
            z_range=layer_config.z_range
        )
        density = phase_solver.add_layer(density, layer)
        print(f"Added layer: {layer}")
    
    # スフィアの設定
    print("\nSetting up spheres...")
    for sphere_config in config.spheres:
        sphere = Sphere(
            phase_name=sphere_config.phase,
            center=sphere_config.center,
            radius=sphere_config.radius
        )
        density = phase_solver.add_sphere(density, sphere)
        print(f"Added sphere: {sphere}")

    return velocity, pressure, density

def simulate(config: SimulationConfig, 
            ns_solver: NavierStokesSolver,
            phase_solver: PhaseFieldSolver,
            fluid_properties: MultiPhaseProperties,
            velocity: list,
            pressure: np.ndarray,
            density: np.ndarray,
            writer: DataWriter):
    """時間発展シミュレーションを実行する
    
    Args:
        config: シミュレーション設定
        ns_solver: Navier-Stokesソルバー
        phase_solver: Phase-fieldソルバー
        fluid_properties: 流体物性
        velocity: 初期速度場
        pressure: 初期圧力場
        density: 初期密度場
        writer: データ出力用ライター
    """
    time = 0.0
    step = 0
    save_count = 0
    
    print("\nStarting simulation...")
    while time < config.max_time:
        # 物性値の更新
        H = phase_solver.heaviside(density)
        current_density = fluid_properties.get_density(H)
        viscosity = fluid_properties.get_viscosity(H)
        
        # 速度場の更新（RK4）
        velocity = ns_solver.runge_kutta4(velocity, current_density, viscosity, config.dt)
        
        # 圧力補正
        velocity, pressure = ns_solver.pressure_projection(velocity, current_density, config.dt)
        
        # 密度場の更新
        density = phase_solver.advance(density, velocity, config.dt)
        
        time += config.dt
        step += 1
        
        # 進捗表示
        if step % 100 == 0:
            print(f"t = {time:.3f}, step = {step}")
            
            # 質量保存のチェック
            total_mass = np.sum(density)
            print(f"Total mass: {total_mass:.6e}")
        
        # 結果の保存
        if step % int(config.save_interval/config.dt) == 0:
            save_count += 1
            print(f"\nSaving output {save_count} at t = {time:.3f}...")
            writer.save_density_field(density, config, f"density_t{time:.3f}")
            
            # 密度の範囲をチェック
            print(f"Density range: [{density.min():.2f}, {density.max():.2f}]")
    
    print("\nSimulation completed!")

def main():
    # 出力ディレクトリの設定
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    writer = DataWriter(output_dir)
    
    # シミュレーションのセットアップ
    print("Setting up simulation...")
    config, ns_solver, phase_solver, fluid_properties = setup_simulation()
    
    # フィールドの初期化
    velocity, pressure, density = initialize_fields(config, phase_solver)
    
    # 初期状態の保存
    print("\nSaving initial state...")
    writer.save_density_field(density, config, f"density_t0.000")
    
    # シミュレーションの実行
    simulate(config, ns_solver, phase_solver, fluid_properties, 
            velocity, pressure, density, writer)

if __name__ == "__main__":
    main()