"""Runge-Kutta 4次法による時間積分を提供するモジュール"""

from typing import Any, Callable, Dict
from ..integrator import TimeIntegratorBase


class RungeKutta4(TimeIntegratorBase):
    """4次のRunge-Kutta法による時間積分"""

    def integrate(
        self, state: Any, dt: float, derivative_fn: Callable[[Any], Any], **kwargs
    ) -> Any:
        """4次のRunge-Kutta法で時間積分を実行

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        k1 = derivative_fn(state, **kwargs)
        k2 = derivative_fn(state + 0.5 * dt * k1, **kwargs)
        k3 = derivative_fn(state + 0.5 * dt * k2, **kwargs)
        k4 = derivative_fn(state + dt * k3, **kwargs)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "method": "Runge-Kutta 4",
            "order": 4,
            "error_estimate": 1 / 30,
            "stability": "無条件安定",
        }
