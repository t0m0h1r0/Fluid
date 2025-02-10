"""時間積分スキームを提供するモジュール

このモジュールは、Navier-Stokes方程式の時間積分に使用される
様々な時間積分スキームを実装します。
"""

from typing import Any, Callable, List

from ..core.interfaces import TimeIntegrator


class ForwardEuler(TimeIntegrator):
    """前進Euler法による時間積分"""

    def step(
        self,
        state: Any,
        dt: float,
        compute_derivative: Callable[[Any, float], Any],
        **kwargs,
    ) -> Any:
        """1時間ステップを実行

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            compute_derivative: 時間微分を計算する関数
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        derivative = compute_derivative(state, **kwargs)
        return state + dt * derivative


class RungeKutta4(TimeIntegrator):
    """4次のRunge-Kutta法による時間積分"""

    def step(
        self,
        state: Any,
        dt: float,
        compute_derivative: Callable[[Any, float], Any],
        **kwargs,
    ) -> Any:
        """1時間ステップを実行

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            compute_derivative: 時間微分を計算する関数
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        # k1の計算
        k1 = compute_derivative(state, **kwargs)

        # k2の計算
        state_k2 = state + 0.5 * dt * k1
        k2 = compute_derivative(state_k2, **kwargs)

        # k3の計算
        state_k3 = state + 0.5 * dt * k2
        k3 = compute_derivative(state_k3, **kwargs)

        # k4の計算
        state_k4 = state + dt * k3
        k4 = compute_derivative(state_k4, **kwargs)

        # 最終的な更新
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class AdamsBashforth(TimeIntegrator):
    """Adams-Bashforth法による時間積分

    2次または3次のAdams-Bashforth法を実装します。
    過去の時間微分を保存して使用します。
    """

    def __init__(self, order: int = 2):
        """Adams-Bashforth法を初期化

        Args:
            order: スキームの次数（2または3）
        """
        if order not in [2, 3]:
            raise ValueError("次数は2または3である必要があります")

        self.order = order
        self._previous_derivatives: List[Any] = []

    def step(
        self,
        state: Any,
        dt: float,
        compute_derivative: Callable[[Any, float], Any],
        **kwargs,
    ) -> Any:
        """1時間ステップを実行

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            compute_derivative: 時間微分を計算する関数
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        # 現在の時間微分を計算
        current_derivative = compute_derivative(state, **kwargs)

        # 過去の微分が不足している場合は前進Euler法を使用
        if len(self._previous_derivatives) < self.order - 1:
            self._previous_derivatives.append(current_derivative)
            return state + dt * current_derivative

        # Adams-Bashforth法の係数
        if self.order == 2:
            coeffs = [3 / 2, -1 / 2]  # 2次のAB法の係数
        else:  # order == 3
            coeffs = [23 / 12, -16 / 12, 5 / 12]  # 3次のAB法の係数

        # 時間発展の計算
        derivatives = [current_derivative] + self._previous_derivatives
        result = state + dt * sum(
            c * d for c, d in zip(coeffs, derivatives[: self.order])
        )

        # 過去の微分を更新
        self._previous_derivatives = [current_derivative] + self._previous_derivatives[
            :-1
        ]

        return result

    def reset(self):
        """積分器の状態をリセット"""
        self._previous_derivatives = []


def create_integrator(integrator_type: str, **kwargs) -> TimeIntegrator:
    """時間積分器を生成

    Args:
        integrator_type: 積分器の種類（"euler", "rk4", "ab2", "ab3"）
        **kwargs: 積分器固有のパラメータ

    Returns:
        生成された時間積分器

    Raises:
        ValueError: 未対応の積分器が指定された場合
    """
    integrators = {
        "euler": ForwardEuler,
        "rk4": RungeKutta4,
        "ab2": lambda: AdamsBashforth(order=2),
        "ab3": lambda: AdamsBashforth(order=3),
    }

    if integrator_type not in integrators:
        raise ValueError(f"未対応の積分器です: {integrator_type}")

    return integrators[integrator_type](**kwargs)
