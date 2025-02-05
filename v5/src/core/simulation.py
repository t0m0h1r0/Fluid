# src/core/simulation.py

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from physics.navier_stokes import NavierStokesSolver
from physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from physics.fluid_properties import MultiPhaseProperties
from .boundary import DirectionalBC
import logging
from datetime import datetime

@dataclass
class SimulationStats:
    """シミュレーション統計情報"""
    timesteps: int = 0
    total_time: float = 0.0
    avg_timestep: float = 0.0
    max_velocity: float = 0.0
    min_density: float = float('inf')
    max_density: float = float('-inf')
    energy_conservation: float = 0.0
    mass_conservation: float = 0.0
    last_update: datetime = datetime.now()

class SimulationManager:
    def __init__(self,
                 config: 'SimulationConfig',
                 navier_stokes: NavierStokesSolver,
                 phase_field: PhaseFieldSolver,
                 fluid_properties: MultiPhaseProperties):
        self.config = config
        self.ns_solver = navier_stokes
        self.phase_solver = phase_field
        self.fluid_properties = fluid_properties
        
        # ロギングの設定
        self.logger = self._setup_logger()
        
        # 統計情報の初期化
        self.stats = SimulationStats()
        
        # 場の初期化
        self.phi = self._initialize_phase_field()
        self.velocity = self._initialize_velocity_field()
        self.pressure = np.zeros(config.shape)
        
        self.time = 0.0
        self.step = 0
        
        # 保存用バッファ
        self.checkpoint_buffer = []
        self.max_checkpoints = 5
    
    def _setup_logger(self) -> logging.Logger:
        """ロギングの設定"""
        logger = logging.getLogger(f"simulation_{datetime.now():%Y%m%d_%H%M%S}")
        logger.setLevel(logging.INFO)
        
        # ファイルハンドラの設定
        fh = logging.FileHandler(f"simulation_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setLevel(logging.INFO)
        
        # フォーマッターの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def advance_timestep(self, adaptive_dt: bool = True) -> Dict[str, np.ndarray]:
        """改良版時間発展"""
        try:
            # 適応的時間刻み幅の計算
            if adaptive_dt:
                dt = self._compute_adaptive_timestep()
            else:
                dt = self.config.dt
            
            # 物性値の更新
            H = self.phase_solver.heaviside(self.phi)
            density = self.fluid_properties.get_density(H)
            viscosity = self.fluid_properties.get_viscosity(H)
            
            # 外力項の計算
            external_forces = self._compute_external_forces(density)
            
            # 速度場の更新（RK4）
            self.velocity = self.ns_solver.runge_kutta4(
                self.velocity,
                density,
                viscosity,
                dt,
                external_forces=external_forces
            )
            
            # 圧力補正
            self.velocity, self.pressure = self.ns_solver.pressure_projection(
                self.velocity,
                density,
                dt
            )
            
            # 相場の更新
            self.phi = self.phase_solver.advance(
                self.phi,
                self.velocity,
                dt
            )
            
            # 統計情報の更新
            self._update_statistics(dt)
            
            self.time += dt
            self.step += 1
            
            # 定期的な検証
            if self.step % 100 == 0:
                self._validate_conservation_laws()
            
            return self._create_output_dict()
            
        except Exception as e:
            self.logger.error(f"Error in timestep {self.step}: {str(e)}")
            raise
    
    def _compute_adaptive_timestep(self) -> float:
        """適応的時間刻み幅の計算"""
        H = self.phase_solver.heaviside(self.phi)
        density = self.fluid_properties.get_density(H)
        viscosity = self.fluid_properties.get_viscosity(H)
        
        dx = min(self.config.Lx/self.config.Nx,
                self.config.Ly/self.config.Ny,
                self.config.Lz/self.config.Nz)
        
        return self.ns_solver.compute_timestep(
            self.velocity,
            density,
            viscosity,
            dx
        )
    
    def _compute_external_forces(self, density: np.ndarray) -> List[np.ndarray]:
        """外力項の計算"""
        gravity = [np.zeros_like(density) for _ in range(3)]
        gravity[2] = -9.81 * density  # z方向の重力
        
        # 表面張力の計算
        surface_tension = self.phase_solver.compute_surface_tension(self.phi)
        
        return [g + st for g, st in zip(gravity, surface_tension)]
    
    def _update_statistics(self, dt: float):
        """統計情報の更新"""
        self.stats.timesteps += 1
        self.stats.total_time += dt
        self.stats.avg_timestep = self.stats.total_time / self.stats.timesteps
        
        # 速度と密度の統計
        H = self.phase_solver.heaviside(self.phi)
        density = self.fluid_properties.get_density(H)
        
        max_vel = max(np.max(np.abs(v)) for v in self.velocity)
        self.stats.max_velocity = max(self.stats.max_velocity, max_vel)
        
        self.stats.min_density = min(self.stats.min_density, np.min(density))
        self.stats.max_density = max(self.stats.max_density, np.max(density))
        
        # エネルギーと質量の保存
        self.stats.energy_conservation = self._compute_energy_conservation()
        self.stats.mass_conservation = self._compute_mass_conservation()
        
        self.stats.last_update = datetime.now()
    
    def _validate_conservation_laws(self):
        """保存則の検証"""
        energy_error = abs(1.0 - self.stats.energy_conservation)
        mass_error = abs(1.0 - self.stats.mass_conservation)
        
        if energy_error > 1e-3:
            self.logger.warning(
                f"Energy conservation violation: {energy_error:.2e}"
            )
        
        if mass_error > 1e-6:
            self.logger.warning(
                f"Mass conservation violation: {mass_error:.2e}"
            )
    
    def _compute_energy_conservation(self) -> float:
        """エネルギー保存の計算"""
        # 運動エネルギー
        kinetic = sum(0.5 * np.sum(v**2) for v in self.velocity)
        # ポテンシャルエネルギー
        H = self.phase_solver.heaviside(self.phi)
        density = self.fluid_properties.get_density(H)
        potential = np.sum(9.81 * density * self.config.Lz)
        
        return (kinetic + potential) / (self.initial_energy + 1e-10)
    
    def _compute_mass_conservation(self) -> float:
        """質量保存の計算"""
        H = self.phase_solver.heaviside(self.phi)
        density = self.fluid_properties.get_density(H)
        return np.sum(density) / (self.initial_mass + 1e-10)
    
    def save_checkpoint(self) -> None:
        """チェックポイントの保存"""
        checkpoint = {
            'time': self.time,
            'step': self.step,
            'fields': {
                'phi': self.phi.copy(),
                'velocity': [v.copy() for v in self.velocity],
                'pressure': self.pressure.copy()
            },
            'stats': self.stats
        }
        
        self.checkpoint_buffer.append(checkpoint)
        if len(self.checkpoint_buffer) > self.max_checkpoints:
            self.checkpoint_buffer.pop(0)
        
        self.logger.info(f"Saved checkpoint at step {self.step}")
    
    def load_checkpoint(self, step: int) -> bool:
        """チェックポイントの読み込み"""
        for checkpoint in self.checkpoint_buffer:
            if checkpoint['step'] == step:
                self.time = checkpoint['time']
                self.step = checkpoint['step']
                self.phi = checkpoint['fields']['phi']
                self.velocity = checkpoint['fields']['velocity']
                self.pressure = checkpoint['fields']['pressure']
                self.stats = checkpoint['stats']
                
                self.logger.info(f"Loaded checkpoint from step {step}")
                return True
        
        self.logger.warning(f"Checkpoint for step {step} not found")
        return False
    
    def _create_output_dict(self) -> Dict[str, np.ndarray]:
        """出力データの作成"""
        return {
            'phi': self.phi,
            'u': self.velocity[0],
            'v': self.velocity[1],
            'w': self.velocity[2],
            'p': self.pressure,
            'stats': self.stats.__dict__
        }