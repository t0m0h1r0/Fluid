"""時間積分スキームを提供するモジュール

このモジュールは、微分方程式の時間積分を行うための数値スキームを提供します。
前進Euler法やRunge-Kutta法など、様々な時間積分手法を実装します。
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class TimeIntegratorBase(ABC):
    """時間積分の基底クラス"""

    @abstractmethod
    def integrate(
        self, state: Any, dt: float, derivative_fn: Callable[[Any], Any], **kwargs
    ) -> Any:
        """時間積分を実行

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {}


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


def create_integrator(integrator_type: str) -> TimeIntegratorBase:
    """時間積分器を生成

    Args:
        integrator_type: 積分器の種類（"euler" または "rk4"）

    Returns:
        生成された時間積分器

    Raises:
        ValueError: 未対応の積分器が指定された場合
    """
    integrators = {
        "euler": ForwardEuler,
        "rk4": RungeKutta4,
    }

    if integrator_type not in integrators:
        raise ValueError(f"未対応の積分器です: {integrator_type}")

    return integrators[integrator_type]()
