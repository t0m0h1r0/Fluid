from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Optional, Tuple, List, TypeVar, Generic

T = TypeVar('T')

class TimeIntegrator(Generic[T], ABC):
    """時間積分法の基底クラス"""
    @abstractmethod
    def step(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> T:
        pass

    @abstractmethod
    def order(self) -> int:
        pass

class RungeKutta4(TimeIntegrator[T]):
    """4次のRunge-Kutta法"""
    def step(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> T:
        k1 = f(t, y)
        k2 = f(t + 0.5*dt, self._add(y, self._mul(k1, 0.5*dt)))
        k3 = f(t + 0.5*dt, self._add(y, self._mul(k2, 0.5*dt)))
        k4 = f(t + dt, self._add(y, self._mul(k3, dt)))
        
        return self._add(y, self._mul(
            self._add(
                self._add(k1, self._mul(k2, 2.0)),
                self._add(self._mul(k3, 2.0), k4)
            ),
            dt/6.0
        ))

    def order(self) -> int:
        return 4

    def _add(self, a: T, b: T) -> T:
        if isinstance(a, np.ndarray):
            return a + b
        elif isinstance(a, list):
            return [x + y for x, y in zip(a, b)]
        else:
            return a + b

    def _mul(self, a: T, b: float) -> T:
        if isinstance(a, np.ndarray):
            return a * b
        elif isinstance(a, list):
            return [x * b for x in a]
        else:
            return a * b

class AdaptiveTimeIntegrator(TimeIntegrator[T]):
    """適応的時間刻み幅制御付き時間積分法"""
    def __init__(self, 
                 base_integrator: TimeIntegrator[T],
                 atol: float = 1e-6,
                 rtol: float = 1e-6,
                 safety: float = 0.9,
                 min_factor: float = 0.2,
                 max_factor: float = 5.0):
        self.integrator = base_integrator
        self.atol = atol
        self.rtol = rtol
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor

    def step(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> Tuple[T, float]:
        """時間発展と次のステップの時間刻み幅を計算"""
        order = self.integrator.order()
        
        # 2つの時間刻み幅での計算
        y1 = self.integrator.step(t, dt, y, f)
        y2 = self.integrator.step(t, dt/2, y, f)
        y2 = self.integrator.step(t + dt/2, dt/2, y2, f)
        
        # 誤差の見積もり
        error = self._estimate_error(y1, y2)
        
        # 許容誤差
        tolerance = self.atol + self.rtol * self._abs(y)
        
        # 新しい時間刻み幅の計算
        factor = self.safety * (tolerance / (error + 1e-15))**(1.0/(order + 1))
        factor = np.clip(factor, self.min_factor, self.max_factor)
        dt_next = dt * factor
        
        # 許容誤差を満たす解の選択
        if error <= tolerance:
            return y2, dt_next  # より高精度な解を採用
        else:
            return y, dt_next * 0.5  # やり直し

    def order(self) -> int:
        return self.integrator.order()

    def _estimate_error(self, y1: T, y2: T) -> float:
        """2つの解の差による誤差推定"""
        diff = self._sub(y1, y2)
        if isinstance(diff, np.ndarray):
            return np.max(np.abs(diff))
        elif isinstance(diff, list):
            return max(np.max(np.abs(d)) for d in diff)
        else:
            return abs(diff)

    def _abs(self, y: T) -> float:
        """状態量の絶対値"""
        if isinstance(y, np.ndarray):
            return np.max(np.abs(y))
        elif isinstance(y, list):
            return max(np.max(np.abs(v)) for v in y)
        else:
            return abs(y)

    def _sub(self, a: T, b: T) -> T:
        """状態量の減算"""
        if isinstance(a, np.ndarray):
            return a - b
        elif isinstance(a, list):
            return [x - y for x, y in zip(a, b)]
        else:
            return a - b

class PredictorCorrector(TimeIntegrator[T]):
    """予測子-修正子法"""
    def __init__(self, predictor: TimeIntegrator[T], corrector: TimeIntegrator[T], iterations: int = 2):
        self.predictor = predictor
        self.corrector = corrector
        self.iterations = iterations

    def step(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> T:
        # 予測子ステップ
        y_pred = self.predictor.step(t, dt, y, f)
        
        # 修正子ステップ
        y_corr = y_pred
        for _ in range(self.iterations):
            y_corr = self.corrector.step(t, dt, y, f)
            
        return y_corr

    def order(self) -> int:
        return min(self.predictor.order(), self.corrector.order())