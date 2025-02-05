from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, TypeVar, Generic

T = TypeVar('T')

class TimeIntegrator(Generic[T], ABC):
    """時間積分法の基底クラス"""
    
    @abstractmethod
    def step(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> T:
        """
        1ステップの時間発展を計算
        
        Args:
            t: 現在の時刻
            dt: 時間刻み幅
            y: 現在の状態
            f: 右辺関数 f(t, y)
        
        Returns:
            次のステップの状態
        """
        pass

    @abstractmethod
    def order(self) -> int:
        """スキームの次数を返す"""
        pass