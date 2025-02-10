from dataclasses import dataclass
import numpy as np
from core.field import VectorField, ScalarField

@dataclass  
class SimulationState:
    time: float
    velocity: VectorField
    levelset: ScalarField 
    pressure: ScalarField

    def get_density(self) -> ScalarField:
        from physics.levelset import heaviside
        return heaviside(self.levelset.data)

    def get_viscosity(self) -> ScalarField:
        from physics.levelset import heaviside 
        mask = heaviside(self.levelset.data)
        return mask * 0.001 + (1 - mask) * 1.5e-5