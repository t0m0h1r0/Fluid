from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from core.field import Field, VectorField
from core.solver import TimeEvolutionSolver
from numerics.poisson_solver import PoissonSolver
from numerics.time_integration import TimeIntegrator, RungeKutta4
from .terms import (ConvectionTerm, DiffusionTerm, PressureTerm, ExternalForcesTerm)
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
    
    def _safe_gradient(self, field: np.ndarray, dx: float, axis: int) -> np.ndarray:
        """安全な勾配計算"""
        grad = np.zeros_like(field)
        slices = [slice(None)] * field.ndim
        
        # 内部点の勾配計算（2次精度中心差分）
        slices[axis] = slice(1, -1)
        grad[tuple(slices)] = (
            field[tuple(slice(2, None) if i == axis else slice(None) 
                       for i in range(field.ndim))] -
            field[tuple(slice(0, -2) if i == axis else slice(None) 
                       for i in range(field.ndim))]
        ) / (2.0 * dx)
        
        # 境界（1次精度片側差分）
        # 左境界
        slices[axis] = 0
        grad[tuple(slices)] = (
            field[tuple(slice(1, 2) if i == axis else slice(None) 
                       for i in range(field.ndim))] -
            field[tuple(slices)]
        ) / dx
        
        # 右境界
        slices[axis] = -1
        grad[tuple(slices)] = (
            field[tuple(slices)] -
            field[tuple(slice(-2, -1) if i == axis else slice(None) 
                       for i in range(field.ndim))]
        ) / dx
        
        return grad
    
    def _safe_divergence(self, velocity: VectorField) -> np.ndarray:
        """安全な発散計算"""
        div = np.zeros_like(velocity.components[0].data)
        
        for i, component in enumerate(velocity.components):
            div += self._safe_gradient(component.data, velocity.dx, i)
        
        return div
    
    def solve(self, velocity: VectorField, dt: float, phi: Field, 
             **kwargs) -> VectorField:
        """1ステップ解く"""
        try:
            # 物性値の取得
            properties = self.fluid_properties.get_mixture_properties(phi)
            
            # 予測子ステップ
            velocity_star = self._predictor_step(velocity, dt, properties)
            
            # 圧力補正ステップ
            pressure = self._pressure_correction_step(velocity_star, dt, properties)
            
            # 修正子ステップ
            velocity_new = self._corrector_step(velocity_star, pressure, dt, properties)
            
            return velocity_new
            
        except Exception as e:
            print(f"Error in NavierStokesSolver.solve: {str(e)}")
            print(f"Velocity shape: {velocity.shape}")
            print(f"Time step: {dt}")
            raise
    
    def _predictor_step(self, velocity: VectorField, dt: float, 
                       properties: Dict[str, np.ndarray]) -> VectorField:
        """予測子ステップ"""
        try:
            density = properties['density']
            viscosity = properties['viscosity']
            
            # 新しい速度場の初期化
            velocity_star = VectorField(velocity.shape, velocity.dx)
            
            for i, component in enumerate(velocity.components):
                # 移流項
                convection = np.zeros_like(component.data)
                for j, v_comp in enumerate(velocity.components):
                    convection -= v_comp.data * self._safe_gradient(
                        component.data, velocity.dx, j
                    )
                
                # 粘性項
                diffusion = np.zeros_like(component.data)
                for j in range(component.data.ndim):
                    grad = self._safe_gradient(component.data, velocity.dx, j)
                    diffusion += self._safe_gradient(grad, velocity.dx, j)
                
                # 重力項（z方向のみ）
                gravity = np.zeros_like(component.data)
                if i == 2:  # z方向
                    gravity.fill(-9.81)
                
                # 時間発展
                velocity_star.components[i].data = component.data + dt * (
                    -convection +
                    viscosity / density * diffusion +
                    gravity
                )
            
            return velocity_star
            
        except Exception as e:
            print(f"Error in predictor step: {str(e)}")
            print(f"Shapes - density: {density.shape}, viscosity: {viscosity.shape}")
            raise
    
    def _pressure_correction_step(self, velocity: VectorField, dt: float,
                                properties: Dict[str, np.ndarray]) -> np.ndarray:
        """圧力補正ステップ"""
        try:
            # 速度の発散を計算
            div_u = self._safe_divergence(velocity)
            
            # 圧力ポアソン方程式の右辺
            rhs = -properties['density'] * div_u / dt
            
            # 圧力を解く
            return self.poisson_solver.solve(rhs)
            
        except Exception as e:
            print(f"Error in pressure correction step: {str(e)}")
            print(f"Divergence shape: {div_u.shape if 'div_u' in locals() else 'not computed'}")
            raise
    
    def _corrector_step(self, velocity: VectorField, pressure: np.ndarray,
                       dt: float, properties: Dict[str, np.ndarray]) -> VectorField:
        """修正子ステップ"""
        try:
            velocity_new = VectorField(velocity.shape, velocity.dx)
            
            for i, component in enumerate(velocity.components):
                # 圧力勾配の計算
                grad_p = self._safe_gradient(pressure, velocity.dx, i)
                
                # 速度の補正
                velocity_new.components[i].data = (
                    component.data - 
                    dt / properties['density'] * grad_p
                )
            
            return velocity_new
            
        except Exception as e:
            print(f"Error in corrector step: {str(e)}")
            print(f"Pressure shape: {pressure.shape}")
            raise
    
    def get_vorticity(self, velocity: VectorField) -> List[np.ndarray]:
        """渦度の計算"""
        try:
            vorticity = []
            
            # x component: ∂w/∂y - ∂v/∂z
            vorticity.append(
                self._safe_gradient(velocity.components[2].data, velocity.dx, 1) -
                self._safe_gradient(velocity.components[1].data, velocity.dx, 2)
            )
            
            # y component: ∂u/∂z - ∂w/∂x
            vorticity.append(
                self._safe_gradient(velocity.components[0].data, velocity.dx, 2) -
                self._safe_gradient(velocity.components[2].data, velocity.dx, 0)
            )
            
            # z component: ∂v/∂x - ∂u/∂y
            vorticity.append(
                self._safe_gradient(velocity.components[1].data, velocity.dx, 0) -
                self._safe_gradient(velocity.components[0].data, velocity.dx, 1)
            )
            
            return vorticity
            
        except Exception as e:
            print(f"Error in vorticity calculation: {str(e)}")
            raise
    
    def get_kinetic_energy(self, velocity: VectorField) -> float:
        """運動エネルギーの計算"""
        try:
            energy = 0.0
            for component in velocity.components:
                energy += np.sum(component.data**2)
            return 0.5 * energy * velocity.dx**3
        except Exception as e:
            print(f"Error in kinetic energy calculation: {str(e)}")
            raise
    
    def get_enstrophy(self, velocity: VectorField) -> float:
        """エンストロフィーの計算"""
        try:
            vorticity = self.get_vorticity(velocity)
            enstrophy = 0.0
            for w in vorticity:
                enstrophy += np.sum(w**2)
            return 0.5 * enstrophy * velocity.dx**3
        except Exception as e:
            print(f"Error in enstrophy calculation: {str(e)}")
            raise
    
    def compute_timestep(self, velocity: VectorField) -> float:
        """安定性条件に基づく時間刻み幅の計算"""
        try:
            # CFL条件
            max_velocity = 0.0
            for component in velocity.components:
                max_velocity = max(max_velocity, np.max(np.abs(component.data)))
            
            dt_convection = (
                float('inf') if max_velocity < 1e-10 
                else self.cfl * velocity.dx / max_velocity
            )
            
            # 粘性制限
            dt_viscous = 0.25 * velocity.dx**2 / (
                np.max(self.fluid_properties.get_viscosity(velocity.components[0])) + 1e-10
            )
            
            # 表面張力制限
            surface_tension = 0.07  # デフォルト値
            min_density = np.min(
                self.fluid_properties.get_density(velocity.components[0])
            )
            dt_surface = np.sqrt(
                min_density * velocity.dx**3 / (2 * np.pi * surface_tension + 1e-10)
            )
            
            # 最小の時間刻み幅を選択
            return min(dt_convection, dt_viscous, dt_surface)
            
        except Exception as e:
            print(f"Error in timestep computation: {str(e)}")
            raise
    
    def get_diagnostics(self, velocity: VectorField) -> Dict[str, float]:
        """診断量の計算"""
        try:
            return {
                'kinetic_energy': self.get_kinetic_energy(velocity),
                'enstrophy': self.get_enstrophy(velocity),
                'max_velocity': max(np.max(np.abs(v.data)) 
                                  for v in velocity.components),
                'divergence_max': np.max(np.abs(self._safe_divergence(velocity)))
            }
        except Exception as e:
            print(f"Error in diagnostics calculation: {str(e)}")
            raise