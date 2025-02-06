from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from .field import Field

class Solver(ABC):
    """ソルバーの基底クラス"""
    
    def __init__(self):
        self._time = 0.0
        self._dt = 0.0
        self._iteration = 0
        
    @property
    def time(self) -> float:
        """現在の時刻"""
        return self._time
    
    @property
    def dt(self) -> float:
        """時間刻み幅"""
        return self._dt
    
    @property
    def iteration(self) -> int:
        """反復回数"""
        return self._iteration
        
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """初期化"""
        pass
    
    @abstractmethod
    def solve(self, field: Field, dt: float) -> Field:
        """1ステップ解く"""
        pass
    
    def update_time(self, dt: float) -> None:
        """時間を更新"""
        self._time += dt
        self._dt = dt
        self._iteration += 1

class TimeEvolutionSolver(Solver):
    """時間発展ソルバーの基底クラス"""
    
    def __init__(self):
        super().__init__()
        self._tolerance = 1e-6
        self._max_iterations = 1000
        
    @property
    def tolerance(self) -> float:
        """収束判定の閾値"""
        return self._tolerance
    
    @tolerance.setter
    def tolerance(self, value: float):
        """収束判定の閾値を設定"""
        if value <= 0:
            raise ValueError("許容誤差は正の値である必要があります")
        self._tolerance = value
        
    @abstractmethod
    def compute_timestep(self, field: Field) -> float:
        """時間刻み幅を計算"""
        pass
    
    @abstractmethod
    def check_convergence(self, field: Field, old_field: Field) -> bool:
        """収束判定"""
        pass

class IterativeSolver(Solver):
    """反復法ソルバーの基底クラス"""
    
    def __init__(self):
        super().__init__()
        self._tolerance = 1e-6
        self._max_iterations = 1000
        self._residual_history = []
        
    @property
    def tolerance(self) -> float:
        """収束判定の閾値"""
        return self._tolerance
    
    @tolerance.setter
    def tolerance(self, value: float):
        if value <= 0:
            raise ValueError("許容誤差は正の値である必要があります")
        self._tolerance = value
        
    @property
    def max_iterations(self) -> int:
        """最大反復回数"""
        return self._max_iterations
    
    @max_iterations.setter
    def max_iterations(self, value: int):
        if value <= 0:
            raise ValueError("最大反復回数は正の整数である必要があります")
        self._max_iterations = value
        
    @abstractmethod
    def compute_residual(self, field: Field) -> float:
        """残差を計算"""
        pass
    
    @abstractmethod
    def iterate(self, field: Field) -> Field:
        """1回の反復"""
        pass
    
    def solve(self, field: Field, dt: float) -> Field:
        """反復法で解く"""
        self._residual_history = []
        iteration = 0
        
        while iteration < self._max_iterations:
            # 1回の反復
            new_field = self.iterate(field)
            
            # 残差の計算と履歴の保存
            residual = self.compute_residual(new_field)
            self._residual_history.append(residual)
            
            # 収束判定
            if residual < self._tolerance:
                return new_field
                
            field = new_field
            iteration += 1
            
        raise RuntimeError(f"収束しませんでした: 残差 = {residual}")
