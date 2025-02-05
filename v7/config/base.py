from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml

@dataclass
class PhaseConfig:
    """各相の物性値設定"""
    name: str
    density: float
    viscosity: float
    surface_tension_coefficient: Optional[float] = None

@dataclass
class DomainConfig:
    """計算領域の設定"""
    Nx: int
    Ny: int 
    Nz: int
    Lx: float
    Ly: float
    Lz: float

    @property
    def resolution(self) -> Tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)

    @property
    def size(self) -> Tuple[float, float, float]:
        return (self.Lx, self.Ly, self.Lz)

@dataclass
class InitialConditionConfig:
    """初期条件の設定"""
    @dataclass
    class Layer:
        phase: str
        z_range: Tuple[float, float]

    @dataclass
    class Sphere:
        center: Tuple[float, float, float]
        radius: float
        phase: str

    layers: List[Layer]
    spheres: List[Sphere]
    initial_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)

@dataclass
class NumericalConfig:
    """数値計算の設定"""
    dt: float
    save_interval: float
    max_time: float
    max_steps: int
    cfl_factor: float
    pressure_tolerance: float
    velocity_tolerance: float

@dataclass
class BoundaryConditionsConfig:
    """境界条件の設定"""
    x: str = 'periodic'
    y: str = 'periodic'
    z: str = 'neumann'

@dataclass
class VisualizationConfig:
    """可視化の設定"""
    @dataclass
    class PlotConfig:
        elev: float
        azim: float

    phase_3d: PlotConfig
    velocity_3d: PlotConfig

@dataclass
class LoggingConfig:
    """ロギングの設定"""
    level: str = 'INFO'
    output_dir: str = 'logs'

@dataclass
class OutputConfig:
    """出力の設定"""
    output_dir: str = "output"
    save_format: str = "hdf5"
    save_interval: float = 0.1
    visualization: bool = True

@dataclass
class PhysicalConfig:
    """物理パラメータの設定"""
    phases: List[PhaseConfig]
    gravity: float = 9.81
    surface_tension: float = 0.07

@dataclass
class Config:
    """シミュレーション全体の設定"""
    physical: PhysicalConfig
    domain: DomainConfig
    initial_condition: InitialConditionConfig
    numerical: NumericalConfig
    boundary_conditions: BoundaryConditionsConfig
    visualization: VisualizationConfig
    logging: LoggingConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # PhaseConfigのリストを作成
        phases = [PhaseConfig(**phase) for phase in data['physical']['phases']]
        
        return cls(
            physical=PhysicalConfig(
                phases=phases,
                gravity=data['physical'].get('gravity', 9.81),
                surface_tension=data['physical'].get('surface_tension', 0.07)
            ),
            domain=DomainConfig(
                Nx=data['domain']['Nx'],
                Ny=data['domain']['Ny'],
                Nz=data['domain']['Nz'],
                Lx=data['domain']['Lx'],
                Ly=data['domain']['Ly'],
                Lz=data['domain']['Lz']
            ),
            initial_condition=InitialConditionConfig(
                layers=[
                    InitialConditionConfig.Layer(**layer) 
                    for layer in data['initial_condition']['layers']
                ],
                spheres=[
                    InitialConditionConfig.Sphere(**sphere) 
                    for sphere in data['initial_condition']['spheres']
                ],
                initial_velocity=tuple(data['initial_condition'].get('initial_velocity', [0.0, 0.0, 0.0]))
            ),
            numerical=NumericalConfig(
                dt=float(data['numerical']['dt']),
                save_interval=float(data['numerical']['save_interval']),
                max_time=float(data['numerical']['max_time']),
                max_steps=int(data['numerical']['max_steps']),
                cfl_factor=float(data['numerical']['cfl_factor']),
                pressure_tolerance=float(data['numerical']['pressure_tolerance']),
                velocity_tolerance=float(data['numerical']['velocity_tolerance'])
            ),
            boundary_conditions=BoundaryConditionsConfig(
                x=data.get('boundary_conditions', {}).get('x', 'periodic'),
                y=data.get('boundary_conditions', {}).get('y', 'periodic'),
                z=data.get('boundary_conditions', {}).get('z', 'neumann')
            ),
            visualization=VisualizationConfig(
                phase_3d=VisualizationConfig.PlotConfig(
                    **data['visualization']['phase_3d']
                ),
                velocity_3d=VisualizationConfig.PlotConfig(
                    **data['visualization']['velocity_3d']
                )
            ),
            logging=LoggingConfig(
                level=data.get('logging', {}).get('level', 'INFO'),
                output_dir=data.get('logging', {}).get('output_dir', 'logs')
            ),
            output=OutputConfig(
                output_dir=data.get('output', {}).get('output_dir', 'output'),
                save_format=data.get('output', {}).get('save_format', 'hdf5'),
                save_interval=data.get('output', {}).get('save_interval', 0.1),
                visualization=data.get('output', {}).get('visualization', True)
            )
        )

    def to_yaml(self, path: Path) -> None:
        """設定をYAMLファイルに出力"""
        data = {
            'physical': {
                'phases': [
                    {
                        'name': phase.name,
                        'density': phase.density,
                        'viscosity': phase.viscosity,
                        'surface_tension_coefficient': phase.surface_tension_coefficient
                    } for phase in self.physical.phases
                ],
                'gravity': self.physical.gravity,
                'surface_tension': self.physical.surface_tension
            },
            'domain': {
                'Nx': self.domain.Nx,
                'Ny': self.domain.Ny,
                'Nz': self.domain.Nz,
                'Lx': self.domain.Lx,
                'Ly': self.domain.Ly,
                'Lz': self.domain.Lz
            },
            'initial_condition': {
                'layers': [
                    {
                        'phase': layer.phase,
                        'z_range': layer.z_range
                    } for layer in self.initial_condition.layers
                ],
                'spheres': [
                    {
                        'center': sphere.center,
                        'radius': sphere.radius,
                        'phase': sphere.phase
                    } for sphere in self.initial_condition.spheres
                ],
                'initial_velocity': list(self.initial_condition.initial_velocity)
            },
            'numerical': {
                'dt': self.numerical.dt,
                'save_interval': self.numerical.save_interval,
                'max_time': self.numerical.max_time,
                'max_steps': self.numerical.max_steps,
                'cfl_factor': self.numerical.cfl_factor,
                'pressure_tolerance': self.numerical.pressure_tolerance,
                'velocity_tolerance': self.numerical.velocity_tolerance
            },
            'boundary_conditions': {
                'x': self.boundary_conditions.x,
                'y': self.boundary_conditions.y,
                'z': self.boundary_conditions.z
            },
            'visualization': {
                'phase_3d': {
                    'elev': self.visualization.phase_3d.elev,
                    'azim': self.visualization.phase_3d.azim
                },
                'velocity_3d': {
                    'elev': self.visualization.velocity_3d.elev,
                    'azim': self.visualization.velocity_3d.azim
                }
            },
            'logging': {
                'level': self.logging.level,
                'output_dir': self.logging.output_dir
            },
            'output': {
                'output_dir': self.output.output_dir,
                'save_format': self.output.save_format,
                'save_interval': self.output.save_interval,
                'visualization': self.output.visualization
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)