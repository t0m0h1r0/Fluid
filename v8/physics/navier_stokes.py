from typing import List, Optional, Dict, Any
import numpy as np
from core.field import Field, VectorField
from core.solver import TimeEvolutionSolver
from .terms import (ConvectionTerm, DiffusionTerm, 
                   PressureTerm, ExternalForcesTerm)
from numerics.poisson_solver import PoissonSolver
from numerics.time_integration import TimeIntegrator, RungeKutta4
from .fluid_properties import FluidProperties

class NavierStokesSolver(TimeEvolutionSolver):
    """ナビエ・ストークスソルバー"""
    
    def __init__(self, 
                 fluid_properties: FluidProperties,
                 poisson_solver: PoissonSolver,
                 time_integrator: Optional[TimeIntegrator] = None):
        """
        Args:
            fluid_properties: 流体物性値
            poisson_solver: 圧力ポアソンソルバー
            time_integrator: 時間積分法
        """
        super().__init__()
        self.fluid_properties = fluid_properties
        self.poisson_solver = poisson_solver
        self.time_integrator = time_integrator or RungeKutta4()
        
        # 各項の初期化
        self.convection = ConvectionTerm(use_weno=True)
        self.diffusion = DiffusionTerm()
        self.pressure = PressureTerm()
        self.external_forces = ExternalForcesTerm()
        
        # 物理パラメータ
        self.cfl = 0.5  # CFL数
        
    def initialize(self, **kwargs):
        """初期化"""
        pass
    
    def solve(self, velocity: VectorField, dt: float, phi: Field, 
             **kwargs) -> VectorField:
        """1ステップ解く"""
        # 物性値の取得
        properties = self.fluid_properties.get_mixture_properties(phi)
        
        # 予測子ステップ
        velocity_star = self._predictor_step(velocity, dt, properties)
        
        # 圧力補正ステップ
        pressure = self._pressure_correction_step(velocity_star, dt, properties)
        
        # 修正子ステップ
        velocity_new = self._corrector_step(velocity_star, pressure, dt, properties)
        
        return velocity_new
    
    def _predictor_step(self, velocity: VectorField, dt: float, 
                       properties: Dict[str, np.ndarray]) -> VectorField:
        """予測子ステップ"""
        def rhs_func(v: VectorField, **kwargs) -> List[np.ndarray]:
            # 各項の計算
            conv = self.convection.compute(v)
            diff = self.diffusion.compute(v, **properties)
            ext = self.external_forces.compute(v, **properties)
            
            # 全項の和
            return [c + d + e for c, d, e in zip(conv, diff, ext)]
        
        # 時間積分
        return self.time_integrator.step(velocity, rhs_func, dt)
    
    def _pressure_correction_step(self, velocity: VectorField, dt: float,
                                properties: Dict[str, np.ndarray]) -> np.ndarray:
        """圧力補正ステップ"""
        # 速度の発散を計算
        div_u = velocity.divergence()
        
        # 圧力ポアソン方程式の右辺
        rhs = -properties['density'] * div_u / dt
        
        # 圧力を解く
        return self.poisson_solver.solve(rhs)
    
    def _corrector_step(self, velocity: VectorField, pressure: np.ndarray,
                       dt: float, properties: Dict[str, np.ndarray]) -> VectorField:
        """修正子ステップ"""
        # 圧力項による補正
        pressure_terms = self.pressure.compute(velocity, pressure=pressure,
                                             **properties)
        
        # 速度の補正
        velocity_new = VectorField(velocity.shape, velocity.dx)
        for i in range(len(velocity.components)):
            velocity_new.components[i].data = (
                velocity.components[i].data + dt * pressure_terms[i]
            )
        
        return velocity_new
    
    def compute_timestep(self, velocity: VectorField) -> float:
        """CFL条件に基づく時間刻み幅の計算"""
        # 対流項によるCFL条件
        max_velocity = max(np.max(np.abs(v.data)) for v in velocity.components)
        dt_convection = self.cfl * velocity.dx / (max_velocity + 1e-10)
        
        # 粘性項による制限
        min_density = np.min(self.fluid_properties.get_density(velocity.components[0]))
        max_viscosity = np.max(self.fluid_properties.get_viscosity(velocity.components[0]))
        dt_diffusion = 0.25 * velocity.dx**2 * min_density / (max_viscosity + 1e-10)
        
        # 表面張力による制限
        surface_tension = 0.07  # デフォルト値
        dt_surface = np.sqrt(
            min_density * velocity.dx**3 / (2 * np.pi * surface_tension + 1e-10)
        )
        
        return min(dt_convection, dt_diffusion, dt_surface)
    
    def check_convergence(self, velocity: VectorField, 
                         old_velocity: VectorField) -> bool:
        """収束判定"""
        max_diff = max(
            np.max(np.abs(v.data - ov.data))
            for v, ov in zip(velocity.components, old_velocity.components)
        )
        return max_diff < self.tolerance
    
    def check_divergence_free(self, velocity: VectorField, 
                            tolerance: float = 1e-6) -> bool:
        """非圧縮条件のチェック"""
        div = velocity.divergence()
        return np.max(np.abs(div)) < tolerance
    
    def get_vorticity(self, velocity: VectorField) -> List[np.ndarray]:
        """渦度の計算"""
        return velocity.curl()
    
    def get_kinetic_energy(self, velocity: VectorField) -> float:
        """運動エネルギーの計算"""
        return 0.5 * sum(
            np.sum(v.data**2) for v in velocity.components
        ) * velocity.dx**3
    
    def get_enstrophy(self, velocity: VectorField) -> float:
        """エンストロフィーの計算"""
        vorticity = self.get_vorticity(velocity)
        return 0.5 * sum(
            np.sum(w**2) for w in vorticity
        ) * velocity.dx**3
    
    def get_strain_rate(self, velocity: VectorField) -> np.ndarray:
        """歪み速度テンソルの計算"""
        strain = np.zeros((3, 3) + velocity.shape)
        
        for i in range(3):
            for j in range(3):
                # ∂ui/∂xj
                strain[i,j] = velocity.components[i].gradient(j)
                
                # 対称化
                if i != j:
                    strain[i,j] += velocity.components[j].gradient(i)
                strain[i,j] *= 0.5
                
        return strain
    
    def get_viscous_dissipation(self, velocity: VectorField) -> float:
        """粘性散逸の計算"""
        strain = self.get_strain_rate(velocity)
        viscosity = self.fluid_properties.get_viscosity(velocity.components[0])
        
        # εij εij の計算
        dissipation = 0.0
        for i in range(3):
            for j in range(3):
                dissipation += np.sum(viscosity * strain[i,j]**2)
        
        return 2.0 * dissipation * velocity.dx**3
    
    def get_diagnostics(self, velocity: VectorField) -> Dict[str, float]:
        """診断量の計算"""
        return {
            'kinetic_energy': self.get_kinetic_energy(velocity),
            'enstrophy': self.get_enstrophy(velocity),
            'viscous_dissipation': self.get_viscous_dissipation(velocity),
            'max_velocity': max(np.max(np.abs(v.data)) for v in velocity.components),
            'divergence_max': np.max(np.abs(velocity.divergence()))
        }
