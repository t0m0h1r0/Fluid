from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml

@dataclass
class DomainConfig:
    """計算領域の設定"""
    size: Tuple[float, float, float]
    resolution: Tuple[int, int, int]
    periodicity: Tuple[bool, bool, bool] = (True, True, False)
    
    @property
    def dx(self) -> Tuple[float, float, float]:
        return tuple(s/n for s, n in zip(self.size, self.resolution))

@dataclass
class FluidConfig:
    """流体の設定"""
    name: str
    density: float
    viscosity: float
    surface_tension: Optional[float] = None
    specific_heat: Optional[float] = None
    thermal_conductivity: Optional[float] = None

@dataclass
class PhaseConfig:
    """相場の設定"""
    epsilon: float = 0.01
    mobility: float = 1.0
    surface_tension: float = 0.07
    stabilization: float = 0.1

@dataclass
class NumericalConfig:
    """数値計算の設定"""
    dt: float
    max_time: float
    cfl_number: float = 0.5
    tolerance: float = 1e-6
    max_iterations: int = 1000

@dataclass
class OutputConfig:
    """出力の設定"""
    save_interval: float
    output_dir: Path
    save_format: str = 'hdf5'
    visualization: bool = True

@dataclass
class Config:
    """シミュレーション設定"""
    domain: DomainConfig
    fluids: Dict[str, FluidConfig]
    phase: PhaseConfig
    numerical: NumericalConfig
    output: OutputConfig
    gravity: float = 9.81

    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            domain=DomainConfig(**data['domain']),
            fluids={name: FluidConfig(name=name, **props) 
                   for name, props in data['fluids'].items()},
            phase=PhaseConfig(**data.get('phase', {})),
            numerical=NumericalConfig(**data['numerical']),
            output=OutputConfig(**data['output']),
            gravity=data.get('gravity', 9.81)
        )

    def to_yaml(self, path: Path) -> None:
        data = {
            'domain': {
                'size': self.domain.size,
                'resolution': self.domain.resolution,
                'periodicity': self.domain.periodicity
            },
            'fluids': {
                name: {
                    'density': fluid.density,
                    'viscosity': fluid.viscosity,
                    'surface_tension': fluid.surface_tension,
                    'specific_heat': fluid.specific_heat,
                    'thermal_conductivity': fluid.thermal_conductivity
                }
                for name, fluid in self.fluids.items()
            },
            'phase': {
                'epsilon': self.phase.epsilon,
                'mobility': self.phase.mobility,
                'surface_tension': self.phase.surface_tension,
                'stabilization': self.phase.stabilization
            },
            'numerical': {
                'dt': self.numerical.dt,
                'max_time': self.numerical.max_time,
                'cfl_number': self.numerical.cfl_number,
                'tolerance': self.numerical.tolerance,
                'max_iterations': self.numerical.max_iterations
            },
            'output': {
                'save_interval': self.output.save_interval,
                'output_dir': str(self.output.output_dir),
                'save_format': self.output.save_format,
                'visualization': self.output.visualization
            },
            'gravity': self.gravity
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)