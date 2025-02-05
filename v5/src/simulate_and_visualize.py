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
    
    density = np.zeros(shape)
    
    # レイヤーの設定
    for layer in config.layers:
        z_min, z_max = layer.z_range
        phase_density = next(p.density for p in config.phases if p.name == layer.phase)
        mask = (Z >= z_min) & (Z < z_max)
        density[mask] = phase_density
    
    # 球の設定
    for sphere in config.spheres:
        phase_density = next(p.density for p in config.phases if p.name == sphere.phase)
        r = np.sqrt(
            (X - sphere.center[0])**2 + 
            (Y - sphere.center[1])**2 + 
            (Z - sphere.center[2])**2
        )
        mask = r <= sphere.radius
        density[mask] = phase_density
    
    return velocity, pressure, density

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
    velocity, pressure, density = initialize_fields(config)
    
    # 初期状態の保存
    print("Saving initial state...")
    writer.save_density_field(density, config, f"density_t0.000")
    
    time = 0.0
    step = 0
    save_count = 0
    
    print("Starting simulation...")
    while time < config.max_time:
        # 時間発展
        velocity = ns_solver.runge_kutta4(velocity, density, config.dt)
        velocity, pressure = ns_solver.pressure_projection(velocity, density, config.dt)
        density = phase_solver.advance(density, velocity, config.dt)
        
        time += config.dt
        step += 1
        
        # 結果の保存
        if step % int(config.save_interval/config.dt) == 0:
            save_count += 1
            print(f"t = {time:.3f}, step = {step}, saving output {save_count}...")
            writer.save_density_field(density, config, f"density_t{time:.3f}")
    
    print("Simulation completed!")

if __name__ == "__main__":
    main()