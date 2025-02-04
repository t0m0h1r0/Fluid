import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physics.navier_stokes import NavierStokesSolver
from src.boundary_conditions.boundary import NSBoundaryManager, BoundaryLocation, NeumannBC
from src.initial_conditions.initial import TwoLayerInitialCondition
from src.utils.visualization import FlowVisualizer

def create_config():
    """Create configuration dictionary for simulation"""
    config = {
        # Grid parameters
        'nx': 32,
        'ny': 32,
        'nz': 64,
        'dx': 1.0/32.,
        'dy': 1.0/32.,
        'dz': 2.0/64.,
        
        # Time stepping parameters
        'dt': 0.001,
        
        # Physical parameters
        'gravity': 9.81,
        
        # Fluid properties
        'fluids': {
            'water': {
                'density': 1000.0,
                'viscosity': 1.0e-3
            },
            'nitrogen': {
                'density': 1.251,
                'viscosity': 1.76e-5
            }
        },
        
        # Two-layer setup parameters
        'bubble_center': [0.5, 0.5, 0.2],
        'bubble_radius': 0.1,
        'layer_height': 1.8,
        
        # Solver parameters
        'order': 8,
        'use_filter': True
    }
    
    return config

def run_simulation(config):
    """Run the two-layer flow simulation"""
    
    # Create solver
    solver = NavierStokesSolver(config)
    
    # Initialize boundary conditions
    bc_manager = NSBoundaryManager()
    bc_manager.set_all_neumann()
    solver.boundary_manager = bc_manager
    
    # Initialize initial conditions
    ic = TwoLayerInitialCondition(config)
    fields = ic.initialize_fields()
    for field_name, field_data in fields.items():
        if hasattr(solver, field_name):
            setattr(solver, field_name, field_data)
    
    # Run simulation
    end_time = 1.0
    solver.solve(end_time)
    
    return solver

if __name__ == "__main__":
    config = create_config()
    solver = run_simulation(config)
    print(f"Simulation completed at t = {solver.get_time():.3f}")
    print(f"Final CFL number: {solver.get_cfl_number():.3f}")

    # 診断情報の取得
    diagnostics = solver.get_diagnostics()
    print(f"Final kinetic energy: {diagnostics['kinetic_energy']}")

    # データの保存
    solver.write_fields('simulation_results.npz')


    visualizer = FlowVisualizer(config)

    # 断面図の表示
    visualizer.plot_slice(solver.phase, plane='xz', title='Phase Field')
    visualizer.plot_slice(solver.w, plane='xz', title='Vertical Velocity')

    # 速度ベクトルの表示
    visualizer.plot_velocity_vectors(solver.u, solver.v, solver.w, plane='xz')
