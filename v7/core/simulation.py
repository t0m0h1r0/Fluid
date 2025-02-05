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
        """初期条件の詳細設定"""
        # まず全体を初期化（デフォルトの相）
        lowest_density_phase = min(
            self.material_manager.fluids, 
            key=lambda p: self.material_manager.fluids[p].density
        )
        initial_value = -1.0 if self.material_manager.fluids[lowest_density_phase].density == min(
            fluid.density for fluid in self.material_manager.fluids.values()
        ) else 1.0

        self.phase.data = np.full(self.phase.data.shape, initial_value)

        # レイヤーの処理
        for layer in self.config.initial_condition.layers:
            # グリッド座標に変換
            z_min, z_max = layer.z_range
            Lz = self.config.domain.Lz
            resolution = self.config.domain.resolution

            # インデックスに変換
            k_min = int(z_min / Lz * resolution[2])
            k_max = int(z_max / Lz * resolution[2])

            # 密度の高い相を1、低い相を-1に設定
            max_density = max(
                self.material_manager.fluids[ph].density for ph in self.material_manager.fluids
            )
            phase_density = self.material_manager.fluids[layer.phase].density

            layer_value = 1.0 if phase_density == max_density else -1.0
            
            # 3次元のスライスを使用
            self.phase.data[:, :, k_min:k_max] = layer_value

        # 球の処理
        for sphere in self.config.initial_condition.spheres:
            # グリッド座標
            X, Y, Z = [
                np.linspace(0, size, resolution[i]) 
                for i, size in enumerate(self.config.domain.size)
            ]
            X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

            # 球の条件
            r = np.sqrt(
                (X - sphere.center[0])**2 + 
                (Y - sphere.center[1])**2 + 
                (Z - sphere.center[2])**2
            )
            mask = r <= sphere.radius

            # 密度の高い相を1、低い相を-1に設定
            max_density = max(
                self.material_manager.fluids[ph].density for ph in self.material_manager.fluids
            )
            phase_density = self.material_manager.fluids[sphere.phase].density

            sphere_value = 1.0 if phase_density == max_density else -1.0
            
            # マスクを適用
            self.phase.data[mask] = sphere_value

        return self.phase.data