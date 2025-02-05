from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Callable, TypeVar, Generic

T = TypeVar('T')

class TimeIntegrator(Generic[T], ABC):
    """時間積分の基底クラス"""
    
    @abstractmethod
    def step(self, y: T, t: float, dt: float, rhs: Callable[[T, float], T]) -> T:
        """1時間ステップの計算
        Args:
            y: 現在の状態
            t: 現在時刻
            dt: 時間刻み幅
            rhs: 右辺の計算関数 f(y, t) -> dy/dt
        Returns:
            次のステップの状態
        """
        pass

    @abstractmethod
    def order(self) -> int:
        """時間積分の次数"""
        pass