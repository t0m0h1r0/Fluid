from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Optional, Tuple, List, TypeVar, Generic, Dict, Union
from core.field.vector_field import VectorField

T = TypeVar('T')

class TimeIntegrator(Generic[T], ABC):
    """時間積分法の基底クラス"""
    @abstractmethod
    def step(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> T:
        """1時間ステップの計算
        
        Args:
            t: 現在時刻
            dt: 時間刻み幅
            y: 現在の状態
            f: 右辺関数 f(t, y)
        
        Returns:
            次のステップの状態
        """
        pass

    @abstractmethod
    def order(self) -> int:
        """スキームの次数"""
        pass

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
        # 段数を設定
        self.stages = len(self.b)
        # 検証
        self._validate_tableau()

    def _validate_tableau(self):
        """ブッチャー表の妥当性検証"""
        if self.a.shape[0] != self.stages or self.a.shape[1] != self.stages:
            raise ValueError("係数行列のサイズが不正です")
        if len(self.c) != self.stages:
            raise ValueError("中間段の時刻の数が不正です")

    def step(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> T:
        """時間発展の1ステップ"""
        if isinstance(y, VectorField):
            return self._step_vector_field(t, dt, y, f)
        else:
            return self._step_generic(t, dt, y, f)

    def _step_vector_field(self, t: float, dt: float, y: VectorField, 
                          f: Callable[[float, VectorField], List[np.ndarray]]) -> VectorField:
        """VectorField用の特殊化されたステップ計算"""
        k = []  # 各段でのベクトル場の変化率
        
        for i in range(self.stages):
            time = t + self.c[i] * dt
            y_stage = y.copy()
            
            for j in range(i):
                # k[j]はリストなので、VectorFieldに変換して演算
                k_field = VectorField(y.metadata)
                k_field.data = k[j]
                y_stage = y_stage + (k_field * (self.a[i, j] * dt))
            
            # f(time, y_stage)の結果をリストで受け取る
            k.append(f(time, y_stage))

        # 最終的な更新
        y_next = y.copy()
        for i in range(self.stages):
            k_field = VectorField(y.metadata)
            k_field.data = k[i]
            y_next = y_next + (k_field * (self.b[i] * dt))
        
        return y_next

    def _step_generic(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> T:
        """一般的な型のための通常のステップ計算"""
        k = []
        for i in range(self.stages):
            time = t + self.c[i] * dt
            y_stage = y
            
            for j in range(i):
                y_stage = y_stage + k[j] * (self.a[i, j] * dt)
            
            k.append(f(time, y_stage))

        y_next = y
        for i in range(self.stages):
            y_next = y_next + k[i] * (self.b[i] * dt)
            
        return y_next

    def order(self) -> int:
        """スキームの次数"""
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