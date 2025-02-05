# core/simulation.py

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from physics.navier_stokes import NavierStokesSolver
from physics.phase_field import PhaseFieldSolver, PhaseFieldParams
from physics.fluid_properties import MultiPhaseProperties
from core.boundary import DirectionalBC
import logging
from datetime import datetime
import os

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
    interface_energy: float = 0.0
    kinetic_energy: float = 0.0
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
        
        # ログディレクトリの作成
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # ロギングの設定
        self.logger = self._setup_logger()
        
        # 統計情報の初期化
        self.stats = SimulationStats()
        
        # 場の初期化
        self.phi = self._initialize_phase_field()
        self.velocity = self._initialize_velocity_field()
        self.pressure = np.zeros((config.Nx, config.Ny, config.Nz))
        
        # 初期エネルギーと質量の記録
        self.initial_energy = self._compute_total_energy()
        self.initial_mass = self._compute_total_mass()
        
        # 時間発展の制御パラメータ
        self.time = 0.0
        self.step = 0
        self.dt_min = 1e-6  # 最小時間刻み
        self.dt_max = 1e-2  # 最大時間刻み
        
        # 収束判定パラメータ
        self.velocity_tolerance = 1e-6
        self.pressure_tolerance = 1e-6
        self.max_iterations = 100

    def _setup_logger(self) -> logging.Logger:
        """ロギングの設定"""
        # ロガーの作成
        logger = logging.getLogger(f"simulation_{datetime.now():%Y%m%d_%H%M%S}")
        logger.setLevel(logging.INFO)
        
        # ファイルハンドラの設定
        log_file = os.path.join(self.log_dir, f"simulation_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # コンソールハンドラの設定
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # フォーマッターの設定
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # ハンドラの追加
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def _initialize_phase_field(self) -> np.ndarray:
        """相場の初期化"""
        # グリッドの作成
        shape = (self.config.Nx, self.config.Ny, self.config.Nz)
        domain_size = (self.config.Lx, self.config.Ly, self.config.Lz)
        
        # 相場の初期化
        phi = self.phase_solver.initialize_field(shape, domain_size)
        
        # レイヤーの設定
        for layer in self.config.layers:
            phi = self.phase_solver.add_layer(phi, layer)
        
        # スフィアの設定
        for sphere in self.config.spheres:
            phi = self.phase_solver.add_sphere(phi, sphere)
        
        return phi

    def _initialize_velocity_field(self) -> List[np.ndarray]:
        """速度場の初期化"""
        shape = (self.config.Nx, self.config.Ny, self.config.Nz)
        return [np.zeros(shape) for _ in range(3)]

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
            
            # 1. 予測子ステップ（速度場の予測）
            velocity_pred = self._predictor_step(density, viscosity, dt)
            
            # 2. 圧力補正
            pressure_new = self._pressure_correction(velocity_pred, density, dt)
            
            # 3. 修正子ステップ（速度場の修正）
            velocity_new = self._corrector_step(velocity_pred, pressure_new, density, dt)
            
            # 4. 相場の更新（Phase-Fieldの時間発展）
            phi_new = self._update_phase_field(velocity_new, dt)
            
            # 収束判定
            if not self._check_convergence(velocity_new, pressure_new):
                self.logger.warning("収束基準を満たしていません")
            
            # フィールドの更新
            self.velocity = velocity_new
            self.pressure = pressure_new
            self.phi = phi_new
            
            # 統計情報の更新
            self._update_statistics(dt)
            
            # 時間の更新
            self.time += dt
            self.step += 1
            
            return self._create_output_dict()
            
        except Exception as e:
            self.logger.error(f"Error in timestep {self.step}: {str(e)}")
            raise

    def _predictor_step(self, 
                       density: np.ndarray, 
                       viscosity: np.ndarray, 
                       dt: float) -> List[np.ndarray]:
        """予測子ステップ：移流拡散方程式の解法"""
        # 外力項の計算（重力と表面張力）
        external_forces = self._compute_external_forces(density)
        
        # RK4による時間積分
        velocity_pred = self.ns_solver.runge_kutta4(
            self.velocity,
            density,
            viscosity,
            dt,
            external_forces=external_forces
        )
        
        return velocity_pred

    def _pressure_correction(self,
                           velocity_pred: List[np.ndarray],
                           density: np.ndarray,
                           dt: float) -> np.ndarray:
        """圧力補正ステップ：圧力ポアソン方程式の解法"""
        # 速度の発散を計算
        div_u = np.zeros_like(density)
        for i, v in enumerate(velocity_pred):
            div_u += self.ns_solver.compute_gradient(v, i)
        
        # 圧力ポアソン方程式のソース項
        rhs = density * div_u / dt
        
        # マルチグリッド法で圧力を解く
        pressure = self.ns_solver.poisson_solver.solve(
            rhs,
            tolerance=self.pressure_tolerance,
            max_iterations=self.max_iterations
        )
        
        return pressure

    def _corrector_step(self,
                       velocity_pred: List[np.ndarray],
                       pressure: np.ndarray,
                       density: np.ndarray,
                       dt: float) -> List[np.ndarray]:
        """修正子ステップ：速度場の補正"""
        velocity_new = []
        for i, v in enumerate(velocity_pred):
            grad_p = self.ns_solver.compute_gradient(pressure, i)
            v_new = v - dt * grad_p / density
            velocity_new.append(v_new)
        
        return velocity_new

    def _update_phase_field(self, 
                           velocity: List[np.ndarray], 
                           dt: float) -> np.ndarray:
        """Phase-Field方程式の時間発展"""
        return self.phase_solver.advance(self.phi, velocity, dt)

    def _compute_adaptive_timestep(self) -> float:
        """CFL条件に基づく適応的時間刻み幅の計算"""
        H = self.phase_solver.heaviside(self.phi)
        density = self.fluid_properties.get_density(H)
        viscosity = self.fluid_properties.get_viscosity(H)
        
        dx = min(self.config.Lx/self.config.Nx,
                self.config.Ly/self.config.Ny,
                self.config.Lz/self.config.Nz)
        
        dt = self.ns_solver.compute_timestep(
            self.velocity,
            density,
            viscosity,
            dx
        )
        
        # 時間刻み幅の制限
        return np.clip(dt, self.dt_min, self.dt_max)

    def _compute_external_forces(self, density: np.ndarray) -> List[np.ndarray]:
        """外力項の計算（重力と表面張力）"""
        # 重力項
        gravity = [np.zeros_like(density) for _ in range(3)]
        gravity[2] = -9.81 * density
        
        # 表面張力の計算
        surface_tension = self.phase_solver.compute_surface_tension(self.phi)
        
        # 外力の合計
        return [g + st for g, st in zip(gravity, surface_tension)]

    def _check_convergence(self,
                          velocity: List[np.ndarray],
                          pressure: np.ndarray) -> bool:
        """収束判定"""
        # 速度の変化
        velocity_change = max(
            np.max(np.abs(v_new - v_old))
            for v_new, v_old in zip(velocity, self.velocity)
        )
        
        # 圧力の変化
        pressure_change = np.max(np.abs(pressure - self.pressure))
        
        return (velocity_change < self.velocity_tolerance and
                pressure_change < self.pressure_tolerance)

    def _compute_total_energy(self) -> float:
        """全エネルギーの計算"""
        # 運動エネルギー
        kinetic = sum(0.5 * np.sum(v**2) for v in self.velocity)
        
        # 界面エネルギー
        interface = self.phase_solver.compute_interface_energy(self.phi)
        
        # ポテンシャルエネルギー
        H = self.phase_solver.heaviside(self.phi)
        density = self.fluid_properties.get_density(H)
        potential = np.sum(9.81 * density * self.config.Lz)
        
        return kinetic + interface + potential

    def _compute_total_mass(self) -> float:
        """全質量の計算"""
        H = self.phase_solver.heaviside(self.phi)
        density = self.fluid_properties.get_density(H)
        return np.sum(density)

    def _update_statistics(self, dt: float):
        """統計情報の更新"""
        self.stats.timesteps += 1
        self.stats.total_time += dt
        self.stats.avg_timestep = self.stats.total_time / self.stats.timesteps
        
        # エネルギーと質量の保存
        current_energy = self._compute_total_energy()
        current_mass = self._compute_total_mass()
        
        self.stats.energy_conservation = current_energy / self.initial_energy
        self.stats.mass_conservation = current_mass / self.initial_mass
        
        # その他の統計情報
        self.stats.max_velocity = max(np.max(np.abs(v)) for v in self.velocity)
        self.stats.interface_energy = self.phase_solver.compute_interface_energy(self.phi)
        self.stats.kinetic_energy = sum(0.5 * np.sum(v**2) for v in self.velocity)
        
        self.stats.last_update = datetime.now()

    def _create_output_dict(self) -> Dict[str, np.ndarray]:
        """出力データの作成"""
        return {
            'phi': self.phi,
            'velocity': self.velocity,
            'pressure': self.pressure,
            'stats': self.stats.__dict__
        }