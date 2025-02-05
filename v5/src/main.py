# main.py
import numpy as np
import os
import yaml
from pathlib import Path
from physics.navier_stokes import NavierStokesSolver
from physics.phase_field import PhaseFieldSolver, PhaseFieldParams, Layer, Sphere
from numerics.compact_scheme import CompactScheme
from numerics.poisson_solver.classic_poisson_solver import ClassicPoissonSolver as PoissonSolver
from core.boundary import DirectionalBC, PeriodicBC, NeumannBC
from physics.fluid_properties import MultiPhaseProperties, FluidProperties
from data_io.data_writer import DataWriter
from utils.config import SimulationConfig

def setup_simulation():
    """シミュレーションの設定とソルバーの初期化"""
    print("シミュレーションの初期化を開始...")
    config = SimulationConfig('config/simulation.yaml')
    
    # スキームと境界条件の設定
    scheme = CompactScheme()
    boundary_conditions = DirectionalBC(
        x_bc=PeriodicBC(),
        y_bc=PeriodicBC(),
        z_bc=NeumannBC()
    )
    
    # ポアソンソルバーの初期化（schemeを追加）
    poisson_solver = PoissonSolver(
        scheme=scheme,
        boundary_conditions=[boundary_conditions.get_condition(i) for i in range(3)]
    )
    
    # 流体物性の設定
    fluids = {
        phase.name: FluidProperties(phase.name, phase.density, phase.viscosity) 
        for phase in config.phases
    }
    fluid_properties = MultiPhaseProperties(fluids)
    
    # ソルバーの初期化
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

def initialize_fields(config, phase_solver):
    """フィールドの初期化"""
    print("\nフィールドの初期化中...")
    
    # 基本フィールドの初期化
    shape = (config.Nx, config.Ny, config.Nz)
    velocity = [np.zeros(shape) for _ in range(3)]
    pressure = np.zeros(shape)
    
    # 密度場の初期化
    density = phase_solver.initialize_field(
        shape=shape,
        domain_size=(config.Lx, config.Ly, config.Lz)
    )
    
    # 相の密度を登録
    print("\n各相の密度を設定:")
    for phase in config.phases:
        phase_solver.set_phase_density(phase.name, phase.density)
        print(f"  {phase.name}: {phase.density} kg/m³")
    
    # レイヤーの設定
    print("\nレイヤーの設定:")
    for layer_config in config.layers:
        layer = Layer(
            phase=layer_config.phase,
            z_range=layer_config.z_range
        )
        density = phase_solver.add_layer(density, layer)
        print(f"  {layer.phase} レイヤー: z = [{layer.z_range[0]}, {layer.z_range[1]}]")
    
    # スフィアの設定
    print("\n球の設定:")
    for sphere_config in config.spheres:
        sphere = Sphere(
            phase=sphere_config.phase,
            center=sphere_config.center,
            radius=sphere_config.radius
        )
        density = phase_solver.add_sphere(density, sphere)
        print(f"  {sphere.phase} 球: 中心 = {sphere.center}, 半径 = {sphere.radius}")
    
    return velocity, pressure, density

def run_simulation():
    """シミュレーションの実行"""
    # 出力ディレクトリの設定
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    writer = DataWriter(str(output_dir))
    
    # シミュレーションの初期化
    config, ns_solver, phase_solver, fluid_properties = setup_simulation()
    velocity, pressure, density = initialize_fields(config, phase_solver)
    
    # 初期状態の保存
    print("\n初期状態を保存中...")
    writer.save_density_field(density, config, "initial_state")
    
    # シミュレーションループ
    print("\nシミュレーション開始...")
    time = 0.0
    step = 0
    save_count = 0
    
    while time < config.max_time:
        # 物性値の更新
        H = phase_solver.heaviside(density)
        current_density = fluid_properties.get_density(H)
        viscosity = fluid_properties.get_viscosity(H)
        
        # 速度場の更新（RK4）
        velocity = ns_solver.runge_kutta4(
            velocity, current_density, viscosity, config.dt
        )
        
        # 圧力補正
        velocity, pressure = ns_solver.pressure_projection(
            velocity, current_density, config.dt
        )
        
        # 密度場の更新
        density = phase_solver.advance(density, velocity, config.dt)
        
        time += config.dt
        step += 1
        
        # 進捗表示
        if step % 10 == 0:
            print(f"\nt = {time:.3f}, ステップ = {step}")
            total_mass = np.sum(density)
            print(f"全質量: {total_mass:.6e}")
            
            # 物理量の範囲をチェック
            print(f"密度範囲: [{density.min():.2f}, {density.max():.2f}]")
            for i, comp in enumerate(['x', 'y', 'z']):
                print(f"{comp}方向速度範囲: [{velocity[i].min():.2e}, {velocity[i].max():.2e}]")
            print(f"圧力範囲: [{pressure.min():.2e}, {pressure.max():.2e}]")
        
        # 結果の保存
        if step % int(config.save_interval/config.dt) == 0:
            save_count += 1
            print(f"\n結果を保存中... (t = {time:.3f})")
            writer.save_density_field(density, config, f"density_t{time:.3f}")
            writer.save_velocity_field(velocity, config, f"velocity_t{time:.3f}")
    
    print("\nシミュレーション完了!")

if __name__ == "__main__":
    run_simulation()