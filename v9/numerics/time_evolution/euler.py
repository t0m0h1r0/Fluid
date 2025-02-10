"""時間積分スキームを提供するモジュール

このモジュールは、異なる時間積分法を実装します。
"""

from typing import Any, Callable, Dict, TypeVar
from .base import TimeIntegrator

StateType = TypeVar("StateType")


class ForwardEuler(TimeIntegrator[StateType]):
    """前進Euler法による時間積分"""

    def integrate(
        self,
        state: StateType,
        dt: float,
        derivative_fn: Callable[[StateType], StateType],
        **kwargs,
    ) -> StateType:
        """前進Euler法で状態を更新

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
            "stability": "条件付き安定",
            "error_estimate": 0.5,
        }


class RungeKutta4(TimeIntegrator[StateType]):
    """4次のRunge-Kutta法による時間積分"""

    def integrate(
        self,
        state: StateType,
        dt: float,
        derivative_fn: Callable[[StateType], StateType],
        **kwargs,
    ) -> StateType:
        """4次のRunge-Kutta法で状態を更新

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        # 各ステージの微分を計算
        k1 = derivative_fn(state, **kwargs)
        k2 = derivative_fn(state + 0.5 * dt * k1, **kwargs)
        k3 = derivative_fn(state + 0.5 * dt * k2, **kwargs)
        k4 = derivative_fn(state + dt * k3, **kwargs)

        # 状態を更新
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "method": "Runge-Kutta 4",
            "order": 4,
            "stability": "無条件安定",
            "error_estimate": 1 / 30,
        }


def create_integrator(method: str = "rk4", **kwargs) -> TimeIntegrator[StateType]:
    """時間積分器を生成

    Args:
        method: 積分法の種類 ('euler', 'rk4')
        **kwargs: 追加のパラメータ

    Returns:
        生成された時間積分器
    """
    integrators = {"euler": ForwardEuler, "rk4": RungeKutta4}

    if method.lower() not in integrators:
        raise ValueError(f"サポートされていない積分法: {method}")

    return integrators[method.lower()](**kwargs)
