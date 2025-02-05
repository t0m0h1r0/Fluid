from dataclasses import dataclass
from typing import Dict, Callable
import numpy as np

@dataclass
class FluidProperties:
    density: float
    viscosity: float
    name: str = ""

class MultiPhaseProperties:
    def __init__(self, fluids: Dict[str, FluidProperties]):
        self.fluids = fluids
        
    def get_density(self, phase_indicator: np.ndarray) -> np.ndarray:
        densities = np.array([fluid.density for fluid in self.fluids.values()])
        return self._interpolate_property(densities, phase_indicator)
    
    def get_viscosity(self, phase_indicator: np.ndarray) -> np.ndarray:
        viscosities = np.array([fluid.viscosity for fluid in self.fluids.values()])
        return self._interpolate_property(viscosities, phase_indicator)
    
    def _interpolate_property(self, properties: np.ndarray, 
                            phase_indicator: np.ndarray,
                            method: str = 'linear') -> np.ndarray:
        if method == 'linear':
            return np.tensordot(properties, phase_indicator, axes=0)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")