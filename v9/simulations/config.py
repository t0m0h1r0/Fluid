from dataclasses import dataclass, field
from typing import Dict

@dataclass
class DomainConfig:
    dimensions: Dict[str, int]
    size: Dict[str, float]

@dataclass 
class PhaseConfig:
    density: float
    viscosity: float

@dataclass
class SolverConfig:
    method: str
    tolerance: float
    max_iterations: int

@dataclass
class OutputConfig:
    directory: str
    format: str
        
@dataclass
class SimulationConfig:
    domain: DomainConfig
    phases: Dict[str, PhaseConfig]
    solver: SolverConfig
    output: OutputConfig
        
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            domain=DomainConfig(**config_dict["domain"]),
            phases={name: PhaseConfig(**phase) for name, phase in config_dict["phases"].items()},
            solver=SolverConfig(**config_dict["solver"]),
            output=OutputConfig(**config_dict["output"]),
        )