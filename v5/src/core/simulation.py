import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from ..physics.navier_stokes import NavierStokesSolver
from ..physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from ..physics.fluid_properties import MultiPhaseProperties
from .boundary import DirectionalBC

@dataclass
class SimulationConfig:
    shape: Tuple[int, ...]
    dt: float
    save_interval: float
    max_time: float
    sphere_center: Tuple[float, ...]
    sphere_radius: float
    top_phase_level: float

class SimulationManager:
    def __init__(self,
                 config: SimulationConfig,
                 navier_stokes: NavierStokesSolver,
                 phase_field: PhaseFieldSolver,
                 fluid_properties: MultiPhaseProperties):
        self.config = config
        self.ns_solver = navier_stokes
        self.phase_solver = phase_field
        self.fluid_properties = fluid_properties
        
        # 場の初期化
        self.phi = self.phase_solver.initialize_field(
            config.shape,
            config.sphere_center,
            config.sphere_radius,
            config.top_phase_level
        )
        self.velocity = [np.zeros(config.shape) for _ in range(3)]
        self.pressure = np.zeros(config.shape)
        
        self.time = 0.0
        self.step = 0
    
    def advance_timestep(self) -> Dict[str, np.ndarray]:
        # 物性値の更新
        H = self.phase_solver.heaviside(self.phi)
        density = self.fluid_properties.get_density(H)
        viscosity = self.fluid_properties.get_viscosity(H)
        
        # 速度場の更新（RK4）
        self.velocity = self.ns_solver.runge_kutta4(
            self.velocity,
            density,
            viscosity,
            self.config.dt
        )
        
        # 圧力補正
        self.velocity, self.pressure = self.ns_solver.pressure_projection(
            self.velocity,
            density,
            self.config.dt
        )
        
        # 相場の更新
        self.phi = self.phase_solver.advance(
            self.phi,
            self.velocity,
            self.config.dt
        )
        
        self.time += self.config.dt
        self.step += 1
        
        return {
            'phi': self.phi,
            'u': self.velocity[0],
            'v': self.velocity[1],
            'w': self.velocity[2],
            'p': self.pressure
        }
    
    def should_save(self) -> bool:
        return abs(self.time / self.config.save_interval - 
                  round(self.time / self.config.save_interval)) < 1e-10
    
    def get_state(self) -> dict:
        return {
            'time': self.time,
            'step': self.step,
            'fields': {
                'phi': self.phi,
                'velocity': self.velocity,
                'pressure': self.pressure
            }
        }
    
    def set_state(self, state: dict):
        self.time = state['time']
        self.step = state['step']
        self.phi = state['fields']['phi']
        self.velocity = state['fields']['velocity']
        self.pressure = state['fields']['pressure']
        
    def validate_state(self) -> Tuple[bool, List[str]]:
        """状態の検証"""
        issues = []
        
        # 質量保存則のチェック
        div_u = sum(self.ns_solver.compute_gradient(v, i) 
                   for i, v in enumerate(self.velocity))
        if np.max(np.abs(div_u)) > 1e-10:
            issues.append(f"Mass conservation violated: max div(u) = {np.max(np.abs(div_u))}")
        
        # 相場の範囲チェック
        if np.any(self.phi < -1) or np.any(self.phi > 1):
            issues.append("Phase field out of range [-1, 1]")
        
        return len(issues) == 0, issues