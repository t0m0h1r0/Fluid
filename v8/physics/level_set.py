import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from core.field import ConservedField
from core.solver import TimeEvolutionSolver
from core.field import VectorField

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
        # 勾配を安全に計算
        grad = np.zeros((self.ndim,) + self.shape)
        for i in range(self.ndim):
            grad[i] = self.safe_gradient(i)
        
        # 勾配の大きさ
        grad_norm = np.sqrt(np.sum(grad ** 2, axis=0))
        grad_norm = np.maximum(grad_norm, self.params.delta_min)
        
        # 正規化された勾配
        grad_normalized = grad / grad_norm[np.newaxis, ...]
        
        # 発散を計算（境界を考慮）
        kappa = np.zeros(self.shape)
        for i in range(self.ndim):
            kappa += self.safe_gradient_divergence(grad_normalized[i], i)
        
        # 曲率を制限
        return np.clip(kappa, -self.params.curvature_cutoff,
                      self.params.curvature_cutoff)
    
    def safe_gradient(self, axis: int) -> np.ndarray:
        """安全な勾配計算（境界を適切に処理）"""
        grad = np.zeros_like(self._data)
        
        # 内部領域の計算（中心差分）
        slices_inner = [slice(1, -1) if i == axis else slice(None) 
                       for i in range(self.ndim)]
        slices_forward = [slice(2, None) if i == axis else slice(None) 
                         for i in range(self.ndim)]
        slices_backward = [slice(0, -2) if i == axis else slice(None) 
                          for i in range(self.ndim)]
        
        grad[tuple(slices_inner)] = (
            self._data[tuple(slices_forward)] -
            self._data[tuple(slices_backward)]
        ) / (2.0 * self.dx)
        
        # 境界の計算（片側差分）
        # 左境界
        slices_left = [slice(0, 1) if i == axis else slice(None) 
                      for i in range(self.ndim)]
        slices_left_next = [slice(1, 2) if i == axis else slice(None) 
                           for i in range(self.ndim)]
        
        grad[tuple(slices_left)] = (
            self._data[tuple(slices_left_next)] -
            self._data[tuple(slices_left)]
        ) / self.dx
        
        # 右境界
        slices_right = [slice(-1, None) if i == axis else slice(None) 
                       for i in range(self.ndim)]
        slices_right_prev = [slice(-2, -1) if i == axis else slice(None) 
                            for i in range(self.ndim)]
        
        grad[tuple(slices_right)] = (
            self._data[tuple(slices_right)] -
            self._data[tuple(slices_right_prev)]
        ) / self.dx
        
        return grad
    
    def safe_gradient_divergence(self, field: np.ndarray, axis: int) -> np.ndarray:
        """安全な勾配の発散計算"""
        div = np.zeros_like(field)
        
        # 内部領域の計算
        slices_inner = [slice(1, -1) if i == axis else slice(None) 
                       for i in range(self.ndim)]
        slices_forward = [slice(2, None) if i == axis else slice(None) 
                         for i in range(self.ndim)]
        slices_backward = [slice(0, -2) if i == axis else slice(None) 
                          for i in range(self.ndim)]
        
        div[tuple(slices_inner)] = (
            field[tuple(slices_forward)] -
            field[tuple(slices_backward)]
        ) / (2.0 * self.dx)
        
        # 境界の計算
        # 左境界
        slices_left = [slice(0, 1) if i == axis else slice(None) 
                      for i in range(self.ndim)]
        slices_left_next = [slice(1, 2) if i == axis else slice(None) 
                           for i in range(self.ndim)]
        
        div[tuple(slices_left)] = (
            field[tuple(slices_left_next)] -
            field[tuple(slices_left)]
        ) / self.dx
        
        # 右境界
        slices_right = [slice(-1, None) if i == axis else slice(None) 
                       for i in range(self.ndim)]
        slices_right_prev = [slice(-2, -1) if i == axis else slice(None) 
                            for i in range(self.ndim)]
        
        div[tuple(slices_right)] = (
            field[tuple(slices_right)] -
            field[tuple(slices_right_prev)]
        ) / self.dx
        
        return div

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
        for _ in range(self.params.reinitialization_steps):
            # 勾配の安全な計算
            grad = np.zeros((self.ndim,) + self.shape)
            for i in range(self.ndim):
                grad[i] = self.safe_gradient(i)
            
            grad_norm = np.sqrt(np.sum(grad ** 2, axis=0))
            
            # 時間発展
            dt = 0.1 * self.dx  # CFLを考慮した時間刻み
            self._data = self._data - dt * np.sign(self._data) * (grad_norm - 1.0)
            
            # 境界付近でスムージング
            self._data = gaussian_filter(self._data, sigma=self.dx)

class LevelSetSolver(TimeEvolutionSolver):
    """Level Set方程式のソルバー"""
    
    def __init__(self, params: Optional[LevelSetParameters] = None):
        super().__init__()
        self.params = params or LevelSetParameters()
    
    def initialize(self, **kwargs) -> None:
        """初期化"""
        pass
    
    def solve(self, field: LevelSetField, dt: float, **kwargs) -> LevelSetField:
        """Level Set方程式を1ステップ解く"""
        if not isinstance(field, LevelSetField):
            raise TypeError("フィールドはLevelSetField型である必要があります")
            
        # 速度場の取得
        velocity = kwargs.get('velocity', None)
        if not isinstance(velocity, VectorField):
            raise ValueError("速度場が正しく指定されていません")
        
        # 新しいフィールドの作成
        new_field = LevelSetField(field.shape, field.dx, field.params)
        
        # 移流項の計算（各方向の勾配を安全に計算）
        advection = np.zeros_like(field.data)
        for i, v in enumerate(velocity.components):
            grad = field.safe_gradient(i)
            advection -= v.data * grad
        
        # 時間発展
        new_field.data = field.data + dt * advection
        
        # リフレッシュが必要な場合は実行
        field._steps_since_refresh += 1
        if field.need_refresh():
            field.refresh()
        
        return new_field
    
    def compute_timestep(self, field: LevelSetField, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算"""
        velocity = kwargs.get('velocity', None)
        if velocity is None:
            raise ValueError("速度場が指定されていません")
        
        # CFL条件に基づく時間刻み幅
        max_velocity = max(np.max(np.abs(v.data)) for v in velocity.components)
        if max_velocity < self.params.delta_min:
            return float('inf')
        
        return 0.5 * field.dx / max_velocity