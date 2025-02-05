from typing import Dict
from pathlib import Path

from config.base import Config
from core.material.properties import MaterialManager
from core.solver.navier_stokes.solver import NavierStokesSolver
from core.solver.time_integrator.base import TimeIntegrator
from physics.phase_field import PhaseField
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class Simulation:
    def __init__(self,
                 material_manager: MaterialManager,
                 phase_field: PhaseField,
                 ns_solver: NavierStokesSolver,
                 time_integrator: TimeIntegrator,
                 config: Config):
        self.material_manager = material_manager
        self.phase_field = phase_field
        self.ns_solver = ns_solver
        self.time_integrator = time_integrator
        self.config = config
        
        # フィールドの初期化
        self.initialize_fields()

    def initialize_fields(self):
        domain_size = self.config.domain.size
        resolution = self.config.domain.resolution
        
        # 相場の初期化
        self.phase = ScalarField(
            name='phase',
            unit='-',
            domain_size=domain_size,
            resolution=resolution
        )
        self.material_manager.initialize_phase_field(self.phase)

        # 速度場の初期化
        self.velocity = VectorField(
            name='velocity',
            unit='m/s',
            domain_size=domain_size,
            resolution=resolution
        )

        # 圧力場の初期化
        self.pressure = ScalarField(
            name='pressure',
            unit='Pa',
            domain_size=domain_size,
            resolution=resolution
        )

    def step(self, dt: float):
        # Phase-Fieldの更新
        self.phase_field.evolve(self.velocity, dt)
        
        # 物性値の更新
        self.material_manager.update_properties(self.phase)
        
        # Navier-Stokes方程式の更新
        fields = {
            'velocity': self.velocity,
            'pressure': self.pressure,
            'density': self.material_manager.get_density_field(),
            'viscosity': self.material_manager.get_viscosity_field()
        }
        
        # 時間発展
        def rhs(t, y):
            return self.ns_solver.compute_right_hand_side(fields)
        
        self.velocity = self.time_integrator.step(0.0, dt, self.velocity, rhs)
        
        # 圧力補正
        self.velocity, self.pressure = self.ns_solver.pressure_correction(
            self.velocity,
            self.material_manager.get_density_field(),
            dt
        )

    def get_field_data(self) -> Dict[str, ScalarField | VectorField]:
        return {
            'phase': self.phase,
            'velocity': self.velocity,
            'pressure': self.pressure,
            'density': self.material_manager.get_density_field(),
            'viscosity': self.material_manager.get_viscosity_field()
        }