import numpy as np
import os 

from physics.navier_stokes import NavierStokesSolver
from physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from numerics.compact_scheme import CompactScheme
from numerics.poisson_solver.multigrid_poisson_solver import MultigridPoissonSolver
from numerics.poisson_solver.classic_poisson_solver import ClassicPoissonSolver
from core.boundary import DirectionalBC, PeriodicBC, NeumannBC
from physics.fluid_properties import MultiPhaseProperties, FluidProperties
from data_io.data_writer import DataWriter
from utils.config import SimulationConfig

def validate_density(density):
    """
    密度場の整合性をチェック
    
    Args:
        density (np.ndarray): 密度場
    
    Returns:
        np.ndarray: 修正された密度場
    """
    # 負の値を0に置換
    density = np.maximum(density, 0)
    
    # ゼロ除算を防ぐため、最小値を設定
    min_density = 1e-10
    density = np.maximum(density, min_density)
    
    # デバッグ情報の出力
    print(f"Density stats: min={density.min()}, max={density.max()}, mean={density.mean()}")
    
    return density

def setup_simulation():
    config = SimulationConfig('config/simulation.yaml')
    scheme = CompactScheme()
    boundary_conditions = DirectionalBC(
        x_bc=PeriodicBC(), y_bc=PeriodicBC(), z_bc=NeumannBC()
    )
    
    # マルチグリッドポアソンソルバー
    poisson_solver = MultigridPoissonSolver(
        scheme=scheme,
        boundary_conditions=[boundary_conditions.get_condition(i) for i in range(3)],
        max_levels=5,           # マルチグリッドの最大レベル数
        smoothing_method='gauss_seidel',  # スムージング手法
        pre_smooth_steps=2,     # 前スムージングステップ数
        post_smooth_steps=2,    # 後スムージングステップ数
        coarse_solver='direct', # 粗いグリッドでの解法
        tolerance=1e-6,         # 収束許容誤差
        max_iterations=100       # 最大反復回数
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
    
    # 密度の安全性を確認して返す
    return velocity, pressure, validate_density(density)

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
    
    # シミュレーションパラメータの初期化
    time = 0.0
    step = 0
    save_count = 0
    
    print("Starting simulation...")
    while time < config.max_time:
        # 密度の安全性を確認
        density = validate_density(density)
        
        # 時間発展
        try:
            velocity = ns_solver.runge_kutta4(velocity, density, config.dt)
            velocity, pressure = ns_solver.pressure_projection(velocity, density, config.dt)
            density = phase_solver.advance(density, velocity, config.dt)
        except Exception as e:
            print(f"Error in simulation step: {e}")
            break
        
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