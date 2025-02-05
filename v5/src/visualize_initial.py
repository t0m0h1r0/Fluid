import numpy as np
from utils.config import SimulationConfig
from data_io.data_writer import DataWriter

def initialize_density_field(config: SimulationConfig) -> np.ndarray:
    shape = (config.Nx, config.Ny, config.Nz)
    density = np.zeros(shape)
    
    x = np.linspace(0, config.Lx, config.Nx)
    y = np.linspace(0, config.Ly, config.Ny)
    z = np.linspace(0, config.Lz, config.Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    for layer in config.layers:
        z_min, z_max = layer.z_range
        phase_density = next(p.density for p in config.phases if p.name == layer.phase)
        mask = (Z >= z_min) & (Z < z_max)
        density[mask] = phase_density
    
    for sphere in config.spheres:
        phase_density = next(p.density for p in config.phases if p.name == sphere.phase)
        r = np.sqrt(
            (X - sphere.center[0])**2 + 
            (Y - sphere.center[1])**2 + 
            (Z - sphere.center[2])**2
        )
        mask = r <= sphere.radius
        density[mask] = phase_density
    
    return density

def main():
    config = SimulationConfig('config/simulation.yaml')
    density = initialize_density_field(config)
    
    writer = DataWriter('output')
    writer.save_density_field(density, config, 'initial_density')

if __name__ == "__main__":
    main()