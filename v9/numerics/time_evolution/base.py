from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic, Dict, Any, Optional

T = TypeVar('T')

class TimeIntegrator(ABC, Generic[T]):
    """時間積分のための抽象基底クラス"""

    def __init__(
        self, 
        stability_limit: float = float("inf"),
        cfl: float = 0.5,  # CFL条件のための係数を追加
        min_dt: float = 1e-6,  # 最小時間刻み幅を追加
        max_dt: float = 1.0,  # 最大時間刻み幅を追加
    ):
        """時間積分器を初期化

        Args:
            stability_limit: 安定性限界
            cfl: CFL数（Courant-Friedrichs-Lewy条件）
            min_dt: 最小時間刻み幅
            max_dt: 最大時間刻み幅
        """
        self._stability_limit = stability_limit
        self._cfl = cfl
        self._min_dt = min_dt
        self._max_dt = max_dt
        self._time = 0.0
        self._dt = None

    @property
    def time(self) -> float:
        """現在の時刻を取得"""
        return self._time

    @property
    def dt(self) -> Optional[float]:
        """時間刻み幅を取得"""
        return self._dt

    @property
    def cfl(self) -> float:
        """CFL数を取得"""
        return self._cfl

    @cfl.setter
    def cfl(self, value: float):
        """CFL数を設定"""
        if value <= 0:
            raise ValueError("CFL数は正の値である必要があります")
        self._cfl = value

    @abstractmethod
    def integrate(self, state: T, dt: float, derivative_fn: Callable[[T], T]) -> T:
        """時間積分を実行する抽象メソッド"""
        pass

    def step_forward(self, dt: float, state: T, **kwargs) -> Dict[str, Any]:
        """時間発展を1ステップ実行

        Args:
            dt: 時間刻み幅
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            時間発展の結果を含む辞書
        """
        # デリバティブを計算する関数を定義
        def derivative_fn(current_state: T) -> T:
            if hasattr(current_state, 'compute_derivative'):
                return current_state.compute_derivative()
            else:
                raise NotImplementedError("状態オブジェクトにcompute_derivativeメソッドが必要です")

        # 時間積分を実行
        new_state = self.integrate(state, dt, derivative_fn)

        # 診断情報の収集
        diagnostics = {
            "time": new_state.time, 
            "dt": dt,
        }

        return {
            "state": new_state,
            "diagnostics": diagnostics
        }

    def check_stability(self, dt: float, state_derivative: T) -> bool:
        """安定性条件をチェック"""
        return dt <= self._stability_limit

    @abstractmethod
    def get_order(self) -> int:
        """積分スキームの次数を返す"""
        pass

    @abstractmethod
    def get_error_estimate(self) -> float:
        """打切り誤差の推定値を返す"""
        pass

    def compute_timestep(self, state: Optional[T] = None, **kwargs) -> float:
        """安定な時間刻み幅を計算

        Args:
            state: 現在の状態（オプション）
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅
        """
        # CFLに基づく時間刻み幅の計算
        if state is not None and hasattr(state, 'velocity'):
            velocity = state.velocity
            max_velocity = max(abs(v.data).max() for v in velocity.components)
            dx = velocity.dx
            dt = self._cfl * dx / (max_velocity + 1e-10)
            
            # 時間刻み幅を制限
            dt = max(min(dt, self._max_dt), self._min_dt)
            return dt
        
        return self._max_dt  # デフォルトの時間刻み幅