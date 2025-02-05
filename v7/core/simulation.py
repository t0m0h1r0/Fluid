from typing import Dict, Optional
import numpy as np

from config.base import Config
from core.material.properties import MaterialManager
from core.solver.navier_stokes.solver import NavierStokesSolver
from core.solver.time_integrator.runge_kutta import TimeIntegrator
from physics.phase_field import PhaseField
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField
from core.field.metadata import FieldMetadata

class MultiPhaseSimulation:
    """多相流体シミュレーションのメインクラス"""
    def __init__(
        self,
        material_manager: MaterialManager,
        phase_field: PhaseField,
        ns_solver: NavierStokesSolver,
        time_integrator: TimeIntegrator,
        config: Config
    ):
        self.material_manager = material_manager
        self.phase_field = phase_field
        self.ns_solver = ns_solver
        self.time_integrator = time_integrator
        self.config = config
        
        # フィールドの初期化
        self.initialize_fields()

    def initialize_fields(self):
        """フィールドの初期化"""
        domain_size = self.config.domain.size
        resolution = self.config.domain.resolution
        
        # 相場の初期化
        phase_metadata = FieldMetadata(
            name='phase',
            unit='-',
            domain_size=domain_size,
            resolution=resolution
        )
        self.phase = self.phase_field.phi
        
        # 速度場の初期化
        velocity_metadata = FieldMetadata(
            name='velocity',
            unit='m/s',
            domain_size=domain_size,
            resolution=resolution
        )
        self.velocity = VectorField(
            velocity_metadata, 
            initial_value=self.config.initial_condition.initial_velocity
        )

        # 圧力場の初期化
        pressure_metadata = FieldMetadata(
            name='pressure',
            unit='Pa',
            domain_size=domain_size,
            resolution=resolution
        )
        self.pressure = ScalarField(pressure_metadata)

        # 密度と粘性の初期化
        self.material_manager.initialize_phase_field(self.phase)

    def step(self, dt: float):
        """1タイムステップの計算"""
        # Phase-Fieldの更新
        self.phase_field.evolve(self.velocity, dt)
        
        # 物性値の更新
        self.material_manager.update_properties(self.phase)
        
        # 各種フィールドの準備
        fields = {
            'velocity': self.velocity,
            'pressure': self.pressure,
            'density': self.material_manager.get_density_field(),
            'viscosity': self.material_manager.get_viscosity_field()
        }
        
        # 時間発展
        def rhs(t: float, y: VectorField) -> VectorField:
            return self.ns_solver.compute_right_hand_side(fields)
        
        # 時間積分
        self.velocity = self.time_integrator.step(0.0, dt, self.velocity, rhs)
        
        # 圧力補正
        self.velocity, self.pressure = self.ns_solver.pressure_correction(
            self.velocity,
            self.material_manager.get_density_field(),
            dt
        )

    def get_field_data(self) -> Dict[str, ScalarField | VectorField]:
        """フィールドデータの取得"""
        return {
            'phase': self.phase,
            'velocity': self.velocity,
            'pressure': self.pressure,
            'density': self.material_manager.get_density_field(),
            'viscosity': self.material_manager.get_viscosity_field()
        }

    def set_initial_conditions(self):
        """初期条件の設定"""
        # 層の設定
        for layer in self.config.initial_condition.layers:
            # TODO: 層に応じて相場を初期化
            z_min, z_max = layer.z_range
            phase_data = self.phase.data
            phase_data[(self.phase.metadata.domain_size[2] * z_min / self.phase.metadata.domain_size[2]):
                       (self.phase.metadata.domain_size[2] * z_max / self.phase.metadata.domain_size[2])] = 1.0
        
        # 球の設定
        for sphere in self.config.initial_condition.spheres:
            # TODO: 球の形状に応じて相場を初期化
            center = sphere.center
            radius = sphere.radius
            
            # 座標のグリッドを生成
            x = np.linspace(0, self.phase.metadata.domain_size[0], self.phase.metadata.resolution[0])
            y = np.linspace(0, self.phase.metadata.domain_size[1], self.phase.metadata.resolution[1])
            z = np.linspace(0, self.phase.metadata.domain_size[2], self.phase.metadata.resolution[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            
            # 球の条件
            sphere_mask = ((X - center[0])**2 + 
                           (Y - center[1])**2 + 
                           (Z - center[2])**2) <= radius**2
            
            self.phase.data[sphere_mask] = 1.0