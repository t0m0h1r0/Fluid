import numpy as np
from typing import List, Dict, Callable, TypeVar, Tuple
from .base import TimeIntegrator

T = TypeVar('T')

class RungeKuttaIntegrator(TimeIntegrator[T]):
    """Runge-Kutta法による時間積分"""
    
    def __init__(self, butcher_tableau: Dict[str, np.ndarray]):
        """
        Args:
            butcher_tableau: ブッチャー表 {'a': array, 'b': array, 'c': array}
                a: 係数行列
                b: 重み係数
                c: 中間段の時刻
        """
        self.a = butcher_tableau['a']
        self.b = butcher_tableau['b']
        self.c = butcher_tableau['c']
        self._validate_tableau()
        self.stages = len(self.b)

    def _validate_tableau(self):
        if self.a.shape[0] != self.stages or self.a.shape[1] != self.stages:
            raise ValueError("係数行列のサイズが不正です")
        if len(self.c) != self.stages:
            raise ValueError("中間段の時刻の数が不正です")

    def step(self, y: T, t: float, dt: float, rhs: Callable[[T, float], T]) -> T:
        k = []
        for i in range(self.stages):
            time = t + self.c[i] * dt
            y_stage = y
            
            for j in range(i):
                y_stage = self._add(y_stage, self._mul(k[j], self.a[i, j] * dt))
            
            k.append(rhs(y_stage, time))

        y_next = y
        for i in range(self.stages):
            y_next = self._add(y_next, self._mul(k[i], self.b[i] * dt))
            
        return y_next

    def _add(self, a: T, b: T) -> T:
        """状態量の加算"""
        if isinstance(a, list):
            return [x + y for x, y in zip(a, b)]
        return a + b

    def _mul(self, a: T, b: float) -> T:
        """状態量のスカラー倍"""
        if isinstance(a, list):
            return [x * b for x in a]
        return a * b

    def order(self) -> int:
        """時間積分の次数"""
        if len(self.b) == 1:
            return 1  # 前進Euler法
        elif len(self.b) == 2:
            return 2  # 2次Runge-Kutta法
        elif len(self.b) == 4:
            return 4  # 4次Runge-Kutta法
        else:
            return 1  # デフォルト

class RK4(RungeKuttaIntegrator[T]):
    """4次Runge-Kutta法"""
    
    def __init__(self):
        # 4次RK法のブッチャー表
        a = np.array([
            [0, 0, 0, 0],
            [1/2, 0, 0, 0],
            [0, 1/2, 0, 0],
            [0, 0, 1, 0]
        ])
        b = np.array([1/6, 1/3, 1/3, 1/6])
        c = np.array([0, 1/2, 1/2, 1])
        
        super().__init__({'a': a, 'b': b, 'c': c})