from typing import Callable, TypeVar, Dict, Any
from .base import TimeIntegrator

T = TypeVar("T")


class ForwardEuler(TimeIntegrator[T]):
    """前進オイラー法による時間積分"""

    def integrate(self, state: T, dt: float, derivative_fn: Callable[[T], T]) -> T:
        """前進オイラー法で状態を更新

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数

        Returns:
            更新された状態
        """
        # 時間微分を計算
        derivative = derivative_fn(state)

        # 状態の各成分を更新
        # velocity成分の更新
        for i, (comp, deriv_comp) in enumerate(
            zip(state.velocity.components, derivative.velocity.components)
        ):
            comp.data += dt * deriv_comp.data

        # levelset成分の更新
        state.levelset.data += dt * derivative.levelset.data

        # 時刻の更新
        state.time += dt

        return state

    def get_order(self) -> int:
        """積分スキームの次数を返す"""
        return 1

    def get_error_estimate(self) -> float:
        """打切り誤差の推定値を返す"""
        return 0.5

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "method": "Forward Euler",
            "order": self.get_order(),
            "stability": "条件付き安定",
            "error_estimate": self.get_error_estimate(),
        }
