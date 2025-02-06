from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from core.field import Field, VectorField
from physics.level_set import LevelSetField
from utils.config import SimulationConfig

@dataclass
class DiagnosticResult:
    """診断結果を保持するクラス"""
    value: float
    status: str  # 'ok', 'warning', 'error'
    message: str

class DiagnosticsAnalyzer:
    """シミュレーションの診断を行うクラス"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self._setup_thresholds()
    
    def _setup_thresholds(self):
        """診断用の閾値設定"""
        self.thresholds = {
            'cfl': {
                'warning': 0.8,
                'error': 1.0
            },
            'divergence': {
                'warning': 1e-4,
                'error': 1e-3
            },
            'energy_conservation': {
                'warning': 1e-3,
                'error': 1e-2
            },
            'mass_conservation': {
                'warning': 1e-5,
                'error': 1e-4
            },
            'velocity_oscillation': {
                'warning': 0.1,
                'error': 0.5
            },
            'interface_thickness': {
                'warning': 2.0,  # グリッド幅の倍数
                'error': 3.0
            }
        }
    
    def run_diagnostics(
        self,
        fields: Dict[str, Any],
        statistics: Dict[str, float]
    ) -> Dict[str, DiagnosticResult]:
        """診断の実行"""
        diagnostics = {}
        
        # CFL条件の診断
        diagnostics['cfl'] = self._check_cfl(fields['velocity'])
        
        # 非圧縮性条件の診断
        diagnostics['divergence'] = self._check_divergence(fields['velocity'])
        
        # 保存則の診断
        if 'mass_conservation' in statistics:
            diagnostics['mass_conservation'] = self._check_mass_conservation(
                statistics['mass_conservation']
            )
        
        if 'energy_conservation' in statistics:
            diagnostics['energy_conservation'] = self._check_energy_conservation(
                statistics['energy_conservation']
            )
        
        # 界面の診断
        diagnostics['interface'] = self._check_interface(fields['phi'])
        
        # 数値的な振動の診断
        diagnostics['oscillation'] = self._check_oscillations(fields)
        
        return diagnostics
    
    def _check_cfl(self, velocity: VectorField) -> DiagnosticResult:
        """CFL条件の診断"""
        max_velocity = max(np.max(np.abs(v.data)) for v in velocity.components)
        cfl = max_velocity * self.config.time.dt / velocity.dx
        
        if cfl > self.thresholds['cfl']['error']:
            return DiagnosticResult(
                value=cfl,
                status='error',
                message='CFL条件を違反しています'
            )
        elif cfl > self.thresholds['cfl']['warning']:
            return DiagnosticResult(
                value=cfl,
                status='warning',
                message='CFL条件が限界に近づいています'
            )
        else:
            return DiagnosticResult(
                value=cfl,
                status='ok',
                message='CFL条件は満たされています'
            )
    
    def _check_divergence(self, velocity: VectorField) -> DiagnosticResult:
        """非圧縮性条件の診断"""
        div = velocity.divergence()
        max_div = np.max(np.abs(div))
        
        if max_div > self.thresholds['divergence']['error']:
            return DiagnosticResult(
                value=max_div,
                status='error',
                message='深刻な発散が検出されました'
            )
        elif max_div > self.thresholds['divergence']['warning']:
            return DiagnosticResult(
                value=max_div,
                status='warning',
                message='発散が大きくなっています'
            )
        else:
            return DiagnosticResult(
                value=max_div,
                status='ok',
                message='発散は許容範囲内です'
            )
    
    def _check_mass_conservation(self, mass_ratio: float) -> DiagnosticResult:
        """質量保存の診断"""
        error = abs(1.0 - mass_ratio)
        
        if error > self.thresholds['mass_conservation']['error']:
            return DiagnosticResult(
                value=error,
                status='error',
                message='質量保存が大きく崩れています'
            )
        elif error > self.thresholds['mass_conservation']['warning']:
            return DiagnosticResult(
                value=error,
                status='warning',
                message='質量保存に乱れが見られます'
            )
        else:
            return DiagnosticResult(
                value=error,
                status='ok',
                message='質量は適切に保存されています'
            )
    
    def _check_energy_conservation(self, energy_ratio: float) -> DiagnosticResult:
        """エネルギー保存の診断"""
        error = abs(1.0 - energy_ratio)
        
        if error > self.thresholds['energy_conservation']['error']:
            return DiagnosticResult(
                value=error,
                status='error',
                message='エネルギー保存が大きく崩れています'
            )
        elif error > self.thresholds['energy_conservation']['warning']:
            return DiagnosticResult(
                value=error,
                status='warning',
                message='エネルギー保存に乱れが見られます'
            )
        else:
            return DiagnosticResult(
                value=error,
                status='ok',
                message='エネルギーは適切に保存されています'
            )
    
    def _check_interface(self, phi: LevelSetField) -> DiagnosticResult:
        """界面の状態診断"""
        # 界面の厚さを評価
        delta = phi.delta()
        thickness = np.mean(delta[delta > phi.params.delta_min]) / phi.dx
        
        if thickness > self.thresholds['interface_thickness']['error']:
            return DiagnosticResult(
                value=thickness,
                status='error',
                message='界面が過度に厚くなっています'
            )
        elif thickness > self.thresholds['interface_thickness']['warning']:
            return DiagnosticResult(
                value=thickness,
                status='warning',
                message='界面が厚くなっています'
            )
        else:
            return DiagnosticResult(
                value=thickness,
                status='ok',
                message='界面の厚さは適切です'
            )
    
    def _check_oscillations(self, fields: Dict[str, Any]) -> DiagnosticResult:
        """数値振動の診断"""
        # 速度場の振動を評価
        velocity = fields['velocity']
        oscillation = self._compute_oscillation_metric(velocity)
        
        if oscillation > self.thresholds['velocity_oscillation']['error']:
            return DiagnosticResult(
                value=oscillation,
                status='error',
                message='深刻な数値振動が検出されました'
            )
        elif oscillation > self.thresholds['velocity_oscillation']['warning']:
            return DiagnosticResult(
                value=oscillation,
                status='warning',
                message='数値振動が見られます'
            )
        else:
            return DiagnosticResult(
                value=oscillation,
                status='ok',
                message='数値振動は許容範囲内です'
            )
    
    def _compute_oscillation_metric(self, velocity: VectorField) -> float:
        """振動の強さを評価する指標を計算"""
        # 2次の中心差分による2階微分を計算
        oscillation_sum = 0.0
        for v in velocity.components:
            d2v = np.zeros_like(v.data)
            for axis in range(v.data.ndim):
                slices_p1 = [slice(None)] * v.data.ndim
                slices_m1 = [slice(None)] * v.data.ndim
                slices_p1[axis] = slice(2, None)
                slices_m1[axis] = slice(0, -2)
                center = slice(1, -1)
                
                d2v[1:-1] += (v.data[tuple(slices_p1)] - 
                             2 * v.data[center] +
                             v.data[tuple(slices_m1)]) / v.dx**2
            
            # 振動の強さを評価
            oscillation_sum += np.mean(np.abs(d2v))
        
        return oscillation_sum / len(velocity.components)
    
    def get_summary(self, diagnostics: Dict[str, DiagnosticResult]) -> str:
        """診断結果のサマリーを生成"""
        summary = "診断結果サマリー:\n"
        
        # 重要度順にメッセージを整理
        for status in ['error', 'warning', 'ok']:
            messages = [
                f"- {key}: {result.message} (値: {result.value:.2e})"
                for key, result in diagnostics.items()
                if result.status == status
            ]
            if messages:
                summary += f"\n{status.upper()}:\n" + "\n".join(messages) + "\n"
        
        return summary
