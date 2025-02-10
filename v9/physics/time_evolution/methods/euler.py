"""前進Euler法による時間積分を提供するモジュール"""

from typing import Any, Callable, Dict
from ..integrator import TimeIntegratorBase


class ForwardEuler(TimeIntegratorBase):
    """前進Euler法による時間積分"""

    def integrate(
        self, state: Any, dt: float, derivative_fn: Callable[[Any], Any], **kwargs
    ) -> Any:
        """前進Euler法で時間積分を実行

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        derivative = derivative_fn(state, **kwargs)
        return state + dt * derivative

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "method": "Forward Euler",
            "order": 1,
            "error_estimate": 0.5,
            "stability": "条件付き安定",
        }
