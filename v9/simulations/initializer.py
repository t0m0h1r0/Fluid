import numpy as np
from .config import SimulationConfig 
from .state import SimulationState
from physics.levelset import LevelSetField

class SimulationInitializer:
    def __init__(self, config: SimulationConfig):
        self.config = config
        
    def create_initial_state(self) -> SimulationState:
        shape = tuple(self.config.domain.dimensions.values())
        
        velocity = np.zeros(shape + (self.config.domain.dimensions["ndim"],))
        levelset = self._initialize_levelset(shape)
        pressure = np.zeros(shape)
        
        return SimulationState(
            time=0.0,
            velocity=velocity,
            levelset=levelset, 
            pressure=pressure,
        )
    
    def _initialize_levelset(self, shape):
        levelset = LevelSetField(shape)
        
        # 水面の高さを設定
        interface_height = 0.5 * self.config.domain.size["z"]
        coords = levelset.get_coordinates()
        levelset.data = interface_height - coords[..., -1] 
        
        return levelset