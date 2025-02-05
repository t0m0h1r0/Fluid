from physics.navier_stokes import NavierStokesSolver
from physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from numerics.compact_scheme import CompactScheme
from numerics.poisson_solver import PoissonSolver
from core.boundary import DirectionalBC, PeriodicBC, NeumannBC
from physics.fluid_properties import MultiPhaseProperties, FluidProperties
from data_io.data_writer import DataWriter
from utils.config import SimulationConfig
import numpy as np
import os 

def setup_simulation():
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

def initialize_fields(config):
    shape = (config.Nx, config.Ny, config.Nz)
    velocity = [np.zeros(shape) for _ in range(3)]
    pressure = np.zeros(shape)
    
    x = np.linspace(0, config.Lx, config.Nx)
    y = np.linspace(0, config.Ly, config.Ny)
    z = np.linspace(0, config.Lz, config.Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 相場の初期化
    phi = np.zeros(shape, dtype=float)
    
    # レイヤーの設定
    for layer in config.layers:
        z_min, z_max = layer.z_range
        phase_name = layer.phase
        phi[Z >= z_min] = 1.0 if phase_name == config.phases[1].name else 0.0
    
    # 球の設定
    for sphere in config.spheres:
        r = np.sqrt(
            (X - sphere.center[0])**2 + 
            (Y - sphere.center[1])**2 + 
            (Z - sphere.center[2])**2
        )
        phase_name = sphere.phase
        phi[r <= sphere.radius] = 1.0 if phase_name == config.phases[1].name else 0.0
    
    return velocity, pressure, phi

def main():
    # 出力ディレクトリの設定
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # シミュレーションのセットアップ
    print("Setting up simulation...")
    config, ns_solver, phase_solver, fluid_properties = setup_simulation()
    writer = DataWriter(output_dir)
    
    # 初期場の設定
    print("Initializing fields...")
    velocity, pressure, phi = initialize_fields(config)
    
    # 初期状態の保存
    print("Saving initial state...")
    writer.save_density_field(phi, config, f"phase_t0.000")
    writer.save_velocity_field(velocity, config, f"phase_t0.000")
    
    time = 0.0
    step = 0
    save_count = 0
    
    print("Starting simulation...")
    while time < config.max_time:
        # 相場の重み関数（ヘビサイド関数）
        H = phase_solver.heaviside(phi)
        
        # 密度と粘度の計算
        density = fluid_properties.get_density(H)
        viscosity = fluid_properties.get_viscosity(H)
        
        # 時間発展
        velocity = ns_solver.runge_kutta4(
            velocity, 
            density, 
            viscosity,  
            config.dt
        )
        velocity, pressure = ns_solver.pressure_projection(velocity, density, config.dt)
        
        # 相場の更新
        phi = phase_solver.advance(phi, velocity, config.dt)
        
        time += config.dt
        step += 1
        
        # 結果の保存
        if step % int(config.save_interval/config.dt) == 0:
            save_count += 1
            print(f"t = {time:.3f}, step = {step}, saving output {save_count}...")
            writer.save_density_field(phi, config, f"phase_t{time:.3f}")
            writer.save_velocity_field(velocity, config, f"phase_t{time:.3f}")
    
    print("Simulation completed!")

if __name__ == "__main__":
    main()