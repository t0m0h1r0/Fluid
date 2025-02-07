from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from core.field import Field, VectorField
from physics.navier_stokes import NavierStokesSolver
from physics.level_set import LevelSetField, LevelSetSolver
from utils.config import SimulationConfig

@dataclass
class TimeStepResult:
    """時間発展の結果を保持するクラス"""
    fields: Dict[str, Any]
    dt: float
    converged: bool = False
    error: Optional[str] = None
    diagnostics: Dict[str, float] = None

class TimeEvolutionManager:
    """時間発展を管理するクラス"""
    
    def __init__(self, config: SimulationConfig):
        """
        Args:
            config: シミュレーション設定
        """
        self.config = config
        self._initialize_time_parameters()
    
    def _initialize_time_parameters(self):
        """時間パラメータの初期化"""
        self.min_dt = 1e-6  # 最小時間刻み幅
        self.max_dt = 1e-2  # 最大時間刻み幅
        self.cfl = self.config.time.cfl
        
        # 安定性パラメータ
        self.stability_params = {
            'viscous_factor': 0.25,  # 粘性項の安定性係数
            'surface_tension_factor': 1.0,  # 表面張力項の安定性係数
            'gravity_factor': 1.0  # 重力項の安定性係数
        }
    
    def advance_timestep(
        self,
        fields: Dict[str, Any],
        solvers: Dict[str, Any]
    ) -> TimeStepResult:
        """1ステップの時間発展
        
        Args:
            fields: フィールドデータ
            solvers: ソルバー群
            
        Returns:
            TimeStepResult: 時間発展の結果
        """
        try:
            # 時間刻み幅の計算
            dt = self._compute_timestep(fields, solvers)
            
            # Level Set方程式の時間発展
            new_phi = self._evolve_level_set(
                fields['phi'],
                fields['velocity'],
                solvers['level_set_solver'],
                dt
            )
            
            # Navier-Stokes方程式の時間発展
            new_velocity, new_pressure = self._evolve_navier_stokes(
                fields['velocity'],
                new_phi,
                solvers['ns_solver'],
                dt
            )
            
            # フィールドの更新
            new_fields = {
                'phi': new_phi,
                'velocity': new_velocity,
                'pressure': new_pressure
            }
            
            # 診断情報の収集
            diagnostics = self._collect_diagnostics(new_fields, solvers, dt)
            
            # 収束判定
            converged = self._check_convergence(fields, new_fields)
            
            return TimeStepResult(
                fields=new_fields,
                dt=dt,
                converged=converged,
                diagnostics=diagnostics
            )
            
        except Exception as e:
            return TimeStepResult(
                fields=fields,
                dt=0.0,
                error=str(e)
            )
    
    def _compute_timestep(
        self,
        fields: Dict[str, Any],
        solvers: Dict[str, Any]
    ) -> float:
        """安定性条件を考慮した時間刻み幅の計算"""
        # CFL条件による制限
        velocity = fields['velocity']
        max_velocity = 0.0
        for component in velocity.components:
            max_velocity = max(max_velocity, np.max(np.abs(component.data)))
            
        if max_velocity < 1e-10:
            dt_convection = self.max_dt
        else:
            dt_convection = self.cfl * velocity.dx / max_velocity
        
        # 粘性項による制限
        dt_viscous = self._compute_viscous_timestep(
            fields['phi'],
            solvers['fluid_properties']
        )
        
        # 表面張力による制限
        dt_surface = self._compute_surface_tension_timestep(
            fields['phi'],
            solvers['fluid_properties']
        )
        
        # 最小の時間刻み幅を選択
        dt = min(dt_convection, dt_viscous, dt_surface)
        
        # 設定された範囲に制限
        return np.clip(dt, self.min_dt, self.max_dt)
    
    def _compute_viscous_timestep(
        self,
        phi: LevelSetField,
        fluid_properties: Any
    ) -> float:
        """粘性項による時間刻み幅の制限"""
        max_viscosity = np.max(fluid_properties.get_viscosity(phi))
        min_density = np.min(fluid_properties.get_density(phi))
        
        if max_viscosity < 1e-10 or min_density < 1e-10:
            return self.max_dt
            
        return self.stability_params['viscous_factor'] * \
               phi.dx**2 * min_density / max_viscosity
    
    def _compute_surface_tension_timestep(
        self,
        phi: LevelSetField,
        fluid_properties: Any
    ) -> float:
        """表面張力による時間刻み幅の制限"""
        min_density = np.min(fluid_properties.get_density(phi))
        # 表面張力係数は0.07 N/m（水-空気）をデフォルト値として使用
        surface_tension = 0.07
        
        if min_density < 1e-10 or surface_tension < 1e-10:
            return self.max_dt
            
        return self.stability_params['surface_tension_factor'] * \
               np.sqrt(min_density * phi.dx**3 / (2 * np.pi * surface_tension))
    
    def _evolve_level_set(
        self,
        phi: LevelSetField,
        velocity: VectorField,
        level_set_solver: LevelSetSolver,
        dt: float
    ) -> LevelSetField:
        """Level Set方程式の時間発展"""
        # 配列形状の一貫性を確保するための修正
        new_phi = level_set_solver.solve(phi, dt, velocity=velocity)
        
        # 必要に応じてリフレッシュ
        if new_phi.need_refresh():
            new_phi.refresh()
        
        return new_phi
    
    def _evolve_navier_stokes(
        self,
        velocity: VectorField,
        phi: LevelSetField,
        ns_solver: NavierStokesSolver,
        dt: float
    ) -> Tuple[VectorField, Field]:
        """Navier-Stokes方程式の時間発展"""
        # 予測子ステップ
        velocity_star = ns_solver.predictor_step(velocity, dt, phi=phi)
        
        # 圧力補正ステップ
        pressure = ns_solver.pressure_correction_step(velocity_star, dt)
        
        # 修正子ステップ
        velocity_new = ns_solver.corrector_step(velocity_star, pressure, dt)
        
        return velocity_new, pressure
    
    def _collect_diagnostics(
        self,
        fields: Dict[str, Any],
        solvers: Dict[str, Any],
        dt: float
    ) -> Dict[str, float]:
        """診断情報の収集"""
        ns_solver = solvers['ns_solver']
        
        return {
            'kinetic_energy': ns_solver.get_kinetic_energy(fields['velocity']),
            'enstrophy': ns_solver.get_enstrophy(fields['velocity']),
            'max_velocity': max(np.max(np.abs(v.data)) 
                              for v in fields['velocity'].components),
            'max_pressure': np.max(np.abs(fields['pressure'].data)),
            'interface_length': np.sum(fields['phi'].delta()),
            'time_step': dt
        }
    
    def _check_convergence(
        self,
        old_fields: Dict[str, Any],
        new_fields: Dict[str, Any]
    ) -> bool:
        """収束判定"""
        # 速度変化による判定
        velocity_change = max(
            np.max(np.abs(new_v.data - old_v.data))
            for new_v, old_v in zip(
                new_fields['velocity'].components,
                old_fields['velocity'].components
            )
        )
        
        # Level Set関数の変化による判定
        phi_change = np.max(np.abs(
            new_fields['phi'].data - old_fields['phi'].data
        ))
        
        return (velocity_change < self.config.solver.velocity_tolerance and
                phi_change < self.config.solver.poisson_tolerance)