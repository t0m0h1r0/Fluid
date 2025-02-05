from dataclasses import dataclass
from typing import List, Tuple
import yaml

@dataclass
class VisualizationConfig:
    phase_3d_elev: float = 30
    phase_3d_azim: float = 45
    velocity_3d_elev: float = 45
    velocity_3d_azim: float = 60

@dataclass
class PhaseConfig:
    name: str
    density: float
    viscosity: float

@dataclass
class SphereConfig:
    center: Tuple[float, float, float]
    radius: float
    phase: str

@dataclass
class LayerConfig:
    phase: str
    z_range: List[float]

class SimulationConfig:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        self.phases = [PhaseConfig(**phase) for phase in config['physical']['phases']]
        self.layers = [LayerConfig(**layer) for layer in config['initial_condition']['layers']]
        self.spheres = [SphereConfig(**sphere) for sphere in config['initial_condition']['spheres']]
        
        self.Nx = config['domain']['Nx']
        self.Ny = config['domain']['Ny']
        self.Nz = config['domain']['Nz']
        self.Lx = config['domain']['Lx']
        self.Ly = config['domain']['Ly']
        self.Lz = config['domain']['Lz']
        
        self.gravity = config['physical']['gravity']
        self.surface_tension = config['physical']['surface_tension']
        
        self.initial_velocity = tuple(config['initial_condition']['initial_velocity'])
        
        self.dt = config['numerical']['dt']
        self.save_interval = config['numerical']['save_interval']
        self.max_time = config['numerical']['max_time']
        
        # 可視化の設定
        vis_config = config.get('visualization', {})
        phase_3d = vis_config.get('phase_3d', {})
        velocity_3d = vis_config.get('velocity_3d', {})
        
        self.visualization = VisualizationConfig(
            phase_3d_elev=phase_3d.get('elev', 30),
            phase_3d_azim=phase_3d.get('azim', 45),
            velocity_3d_elev=velocity_3d.get('elev', 45),
            velocity_3d_azim=velocity_3d.get('azim', 60)
        )