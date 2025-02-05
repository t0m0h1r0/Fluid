import numpy as np
from typing import Callable, Tuple, TypeVar, Generic

from .base import TimeIntegrator

T = TypeVar('T')

class AdaptiveTimeIntegrator(TimeIntegrator[T]):
    """適応的時間刻み幅制御付き時間積分法"""
    def __init__(
        self, 
        base_integrator: TimeIntegrator[T],
        atol: float = 1e-6,
        rtol: float = 1e-6,
        safety: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0
    ):
        """
        Args:
            base_integrator: ベースの時間積分器
            atol: 絶対許容誤差
            rtol: 相対許容誤差
            safety: 安全係数
            min_factor: 最小時間刻み幅係数
            max_factor: 最大時間刻み幅係数
        """
        self.integrator = base_integrator
        self.atol = atol
        self.rtol = rtol
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor

    def step(self, t: float, dt: float, y: T, f: Callable[[float, T], T]) -> T:
        """
        適応的時間発展
        
        Args:
            t: 現在時刻
            dt: 初期時間刻み幅
            y: 現在の状態
            f: 右辺関数
        
        Returns:
            次のステップの状態
        """
        order = self.order()
        
        # 2つの異なる時間刻み幅での計算
        y1 = self.integrator.step(t, dt, y, f)
        y2 = self.integrator.step(t, dt/2, y, f)
        y2 = self.integrator.step(t + dt/2, dt/2, y2, f)
        
        # 誤差の推定
        error = self._estimate_error(y1, y2)
        
        # 許容誤差の計算
        abs_y = float(self._abs(y))  # 明示的にfloatに変換
        tolerance = float(self.atol) + float(self.rtol) * abs_y
        
        # 新しい時間刻み幅の計算
        factor = self.safety * (tolerance / (error + 1e-15))**(1.0 / (order + 1))
        factor = np.clip(factor, self.min_factor, self.max_factor)
        dt_next = dt * factor
        
        # 許容誤差を満たす解の選択
        if error <= tolerance:
            return y2  # より高精度な解を採用
        else:
            return y  # 時間刻み幅を小さくして再計算が必要

    def order(self) -> int:
        """
        時間積分スキームの次数を返す
        
        Returns:
            スキームの次数
        """
        return self.integrator.order()

    def _estimate_error(self, y1, y2):
        """
        2つの解の差による誤差推定
        
        Args:
            y1: 1つ目の解
            y2: 2つ目の解
        
        Returns:
            誤差の大きさ
        """
        from core.field.vector_field import VectorField
        from core.field.scalar_field import ScalarField
        
        # VectorFieldの場合
        if isinstance(y1, VectorField):
            # 各成分の最大絶対誤差を計算
            diff_components = [np.abs(c1 - c2) for c1, c2 in zip(y1.data, y2.data)]
            return max(np.max(diff) for diff in diff_components)
        
        # ScalarFieldの場合
        if isinstance(y1, ScalarField):
            return np.max(np.abs(y1.data - y2.data))
        
        # それ以外（numpy配列など）
        if isinstance(y1, np.ndarray):
            return np.max(np.abs(y1 - y2))
        
        # リストの場合
        if isinstance(y1, list):
            return max(np.max(np.abs(np.array(y1) - np.array(y2))))
        
        # 単純な値の場合
        return abs(y1 - y2)

    def _abs(self, y):
        """
        状態量の絶対値の計算
        
        Args:
            y: 状態量
        
        Returns:
            絶対値の最大値（スカラー値）
        """
        from core.field.vector_field import VectorField
        from core.field.scalar_field import ScalarField
        
        # VectorFieldの場合
        if isinstance(y, VectorField):
            # 各成分の最大絶対値を計算し、その最大値を返す
            max_abs_values = [np.max(np.abs(component)) for component in y.data]
            return float(np.max(max_abs_values))
        
        # ScalarFieldの場合
        if isinstance(y, ScalarField):
            return float(np.max(np.abs(y.data)))
        
        # numpy配列の場合
        if isinstance(y, np.ndarray):
            return float(np.max(np.abs(y)))
        
        # リストの場合
        if isinstance(y, list):
            return float(max(np.max(np.abs(np.array(item))) for item in y))
        
        # 単純な値の場合
        return float(abs(y))

    def _sub(self, a, b):
        """
        状態量の減算
        
        Args:
            a: 被減算量
            b: 減算量
        
        Returns:
            差
        """
        from core.field.vector_field import VectorField
        from core.field.scalar_field import ScalarField
        
        # VectorFieldの場合
        if isinstance(a, VectorField):
            # 各成分の減算
            return type(a)(a.metadata, 
                [c1 - c2 for c1, c2 in zip(a.data, b.data)])
        
        # ScalarFieldの場合
        if isinstance(a, ScalarField):
            return type(a)(a.metadata, a.data - b.data)
        
        # numpy配列の場合
        if isinstance(a, np.ndarray):
            return a - b
        
        # リストの場合
        if isinstance(a, list):
            return [x - y for x, y in zip(a, b)]
        
        # 単純な値の場合
        return a - b