from typing import Callable, TypeVar, Dict, Any
from .base import TimeIntegrator

T = TypeVar("T")


class RungeKutta4(TimeIntegrator[T]):
    """4次のRunge-Kutta法による時間積分"""

    def integrate(self, state: T, dt: float, derivative_fn: Callable[[T], T]) -> T:
        """4次のRunge-Kutta法で状態を更新

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数

        Returns:
            更新された状態
        """
        # 各ステージの微分を計算
        k1 = derivative_fn(state)
        k2 = derivative_fn(state + 0.5 * dt * k1)
        k3 = derivative_fn(state + 0.5 * dt * k2)
        k4 = derivative_fn(state + dt * k3)

        # 状態を更新
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def get_order(self) -> int:
        """積分スキームの次数を返す"""
        return 4

    def get_error_estimate(self) -> float:
        """打切り誤差の推定値を返す"""
        return 1 / 30

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "method": "Runge-Kutta 4",
            "order": self.get_order(),
            "stability": "無条件安定",
            "error_estimate": self.get_error_estimate(),
        }
