import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from core.field import ConservedField
from core.solver import TimeEvolutionSolver

@dataclass
class LevelSetParameters:
    """Level Set法のパラメータ"""
    epsilon: float = 1.0e-2  # 界面の厚さ
    delta_min: float = 1.0e-10  # デルタ関数の最小値
    refresh_interval: int = 1  # リフレッシュ間隔
    reinitialization_steps: int = 5  # 再初期化のステップ数
    curvature_cutoff: float = 100.0  # 曲率の閾値

class LevelSetField(ConservedField):
    """Level Set法による界面表現"""
    
    def __init__(self, shape: Tuple[int, ...], dx: float = 1.0,
                 params: Optional[LevelSetParameters] = None):
        super().__init__(shape, dx)
        self.params = params or LevelSetParameters()
        self._steps_since_refresh = 0
    
    def heaviside(self) -> np.ndarray:
        """ヘビサイド関数"""
        return 0.5 * (1.0 + np.tanh(self._data / self.params.epsilon))
    
    def delta(self) -> np.ndarray:
        """デルタ関数"""
        # 正規化されたデルタ関数
        delta = (1.0 / (2.0 * self.params.epsilon)) * (
            1.0 - np.tanh(self._data / self.params.epsilon) ** 2
        )
        
        # 最小値でクリップ
        return np.maximum(delta, self.params.delta_min)
    
    def curvature(self) -> np.ndarray:
        """界面の曲率を計算"""
        # 勾配を計算
        grad = np.array([self.gradient(i) for i in range(self.ndim)])
        
        # 勾配の大きさ
        grad_norm = np.sqrt(np.sum(grad ** 2, axis=0))
        grad_norm = np.maximum(grad_norm, self.params.delta_min)
        
        # 正規化された勾配
        grad_normalized = grad / grad_norm
        
        # 発散を計算
        kappa = sum(np.gradient(grad_normalized[i], self.dx, axis=i)
                   for i in range(self.ndim))
        
        # 曲率を制限
        return np.clip(kappa, -self.params.curvature_cutoff,
                      self.params.curvature_cutoff)
    
    def need_refresh(self) -> bool:
        """リフレッシュが必要かどうかを判定"""
        return (self._steps_since_refresh >= 
                self.params.refresh_interval)
    
    def refresh(self):
        """Level Set関数のリフレッシュ"""
        if not self.need_refresh():
            return
            
        # 初期の質量を記録
        initial_mass = self.integrate()
        
        # 再初期化
        self._reinitialize()
        
        # 質量を保存するように補正
        current_mass = self.integrate()
        if abs(current_mass) > self.params.delta_min:
            self._data *= np.sqrt(initial_mass / current_mass)
        
        self._steps_since_refresh = 0
    
    def _reinitialize(self):
        """Level Set関数の再初期化"""
        # 符号付き距離関数に変換
        for _ in range(self.params.reinitialization_steps):
            # 勾配の計算
            grad = np.array([self.gradient(i) for i in range(self.ndim)])
            grad_norm = np.sqrt(np.sum(grad ** 2, axis=0))
            
            # 時間発展
            dt = 0.1 * self.dx  # CFLを考慮した時間刻み
            self._data = self._data - dt * np.sign(self._data) * (grad_norm - 1.0)
            
            # 境界付近でスムージング
            self._data = gaussian_filter(self._data, sigma=self.dx)

    def reinitialize(self):
        """外部から呼び出す再初期化メソッド"""
        self._reinitialize()