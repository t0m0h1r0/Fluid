from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from core.field import Field, VectorField
from physics.level_set import LevelSetField
from utils.config import SimulationConfig

@dataclass
class SimulationStats:
    """シミュレーションの統計情報を保持するクラス"""
    time_step_history: List[float] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    mass_history: List[float] = field(default_factory=list)
    max_velocity_history: List[float] = field(default_factory=list)
    interface_length_history: List[float] = field(default_factory=list)
    enstrophy_history: List[float] = field(default_factory=list)
    pressure_peaks: List[float] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)

class StatisticsAnalyzer:
    """シミュレーションの統計解析を行うクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.stats = SimulationStats()
        self.initial_values = {}
    
    def initialize(self, fields: Dict[str, Any]):
        """初期値の記録"""
        self.initial_values = {
            'mass': self._compute_mass(fields['phi']),
            'energy': self._compute_kinetic_energy(fields['velocity']),
            'interface_length': self._compute_interface_length(fields['phi'])
        }
    
    def update(self, fields: Dict[str, Any], dt: float):
        """統計情報の更新"""
        # 基本的な統計量の記録
        self.stats.time_step_history.append(dt)
        self.stats.energy_history.append(
            self._compute_kinetic_energy(fields['velocity'])
        )
        self.stats.mass_history.append(
            self._compute_mass(fields['phi'])
        )
        self.stats.max_velocity_history.append(
            self._compute_max_velocity(fields['velocity'])
        )
        self.stats.interface_length_history.append(
            self._compute_interface_length(fields['phi'])
        )
        self.stats.enstrophy_history.append(
            self._compute_enstrophy(fields['velocity'])
        )
        self.stats.pressure_peaks.append(
            np.max(np.abs(fields['pressure'].data))
        )
        
        # 収束性の評価
        if len(self.stats.energy_history) > 1:
            convergence = abs(
                self.stats.energy_history[-1] - self.stats.energy_history[-2]
            ) / self.stats.energy_history[-2]
            self.stats.convergence_history.append(convergence)
    
    def get_current_stats(self) -> Dict[str, float]:
        """現在の統計情報を取得"""
        if not self._has_history():
            return {}
            
        return {
            'time_step': self.stats.time_step_history[-1],
            'energy': self.stats.energy_history[-1],
            'mass': self.stats.mass_history[-1],
            'max_velocity': self.stats.max_velocity_history[-1],
            'interface_length': self.stats.interface_length_history[-1],
            'enstrophy': self.stats.enstrophy_history[-1],
            'mass_conservation': self._compute_mass_conservation(),
            'energy_conservation': self._compute_energy_conservation()
        }
    
    def get_conservation_errors(self) -> Dict[str, float]:
        """保存則の誤差を取得"""
        return {
            'mass_error': self._compute_mass_conservation_error(),
            'energy_error': self._compute_energy_conservation_error(),
        }
    
    def get_convergence_metrics(self) -> Dict[str, float]:
        """収束性の指標を取得"""
        if not self._has_history():
            return {}
            
        return {
            'energy_convergence': self.stats.convergence_history[-1],
            'max_velocity': self.stats.max_velocity_history[-1],
            'pressure_oscillation': self._compute_pressure_oscillation()
        }
    
    def _compute_kinetic_energy(self, velocity: VectorField) -> float:
        """運動エネルギーの計算"""
        return 0.5 * sum(
            np.sum(v.data**2) for v in velocity.components
        ) * velocity.dx**3
    
    def _compute_mass(self, phi: LevelSetField) -> float:
        """質量の計算"""
        return np.sum(phi.heaviside()) * phi.dx**3
    
    def _compute_interface_length(self, phi: LevelSetField) -> float:
        """界面長さの計算"""
        return np.sum(phi.delta()) * phi.dx**2
    
    def _compute_max_velocity(self, velocity: VectorField) -> float:
        """最大速度の計算"""
        return max(np.max(np.abs(v.data)) for v in velocity.components)
    
    def _compute_enstrophy(self, velocity: VectorField) -> float:
        """エンストロフィーの計算"""
        curl = velocity.curl()
        return 0.5 * sum(np.sum(w**2) for w in curl) * velocity.dx**3
    
    def _compute_mass_conservation(self) -> float:
        """質量保存の評価"""
        if not self._has_history() or not self.initial_values:
            return 1.0
        return self.stats.mass_history[-1] / self.initial_values['mass']
    
    def _compute_energy_conservation(self) -> float:
        """エネルギー保存の評価"""
        if not self._has_history() or not self.initial_values:
            return 1.0
        return self.stats.energy_history[-1] / self.initial_values['energy']
    
    def _compute_mass_conservation_error(self) -> float:
        """質量保存の誤差"""
        return abs(1.0 - self._compute_mass_conservation())
    
    def _compute_energy_conservation_error(self) -> float:
        """エネルギー保存の誤差"""
        return abs(1.0 - self._compute_energy_conservation())
    
    def _compute_pressure_oscillation(self) -> float:
        """圧力振動の評価"""
        if len(self.stats.pressure_peaks) < 3:
            return 0.0
        
        # 最近の圧力ピーク値の変動を評価
        recent_peaks = self.stats.pressure_peaks[-3:]
        return np.std(recent_peaks) / np.mean(recent_peaks)
    
    def _has_history(self) -> bool:
        """履歴が存在するかチェック"""
        return len(self.stats.time_step_history) > 0
    
    def get_time_averages(self, window: Optional[int] = None) -> Dict[str, float]:
        """時間平均を計算"""
        if not self._has_history():
            return {}
            
        if window is None:
            window = len(self.stats.time_step_history)
        
        window = min(window, len(self.stats.time_step_history))
        
        return {
            'avg_energy': np.mean(self.stats.energy_history[-window:]),
            'avg_enstrophy': np.mean(self.stats.enstrophy_history[-window:]),
            'avg_max_velocity': np.mean(self.stats.max_velocity_history[-window:]),
            'avg_interface_length': np.mean(
                self.stats.interface_length_history[-window:]
            )
        }
