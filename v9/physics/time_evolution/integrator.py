"""時間積分スキームを提供するモジュール

このモジュールは、微分方程式の時間積分を行うための数値スキームを実装します。
前進Euler法やRunge-Kutta法など、様々な時間積分手法を実装します。
"""

from typing import Any, Callable, Dict, Protocol, Union


class TimeIntegratorBase(Protocol):
    """時間積分のプロトコル"""

    def integrate(
        self,
        dt: float,
        derivative_fn: Callable[[Any, float], Any],
        state: Any,
        **kwargs,
    ) -> Any:
        """時間積分を実行

        Args:
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


class ForwardEuler:
    """前進Euler法による時間積分"""

    def integrate(
        self,
        dt: float,
        derivative_fn: Callable[[Any, float], Any],
        state: Any,
        **kwargs,
    ) -> Any:
        """前進Euler法で時間積分を実行

        Args:
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        derivative = derivative_fn(state, **kwargs)
        return state + dt * derivative

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {"method": "Forward Euler", "order": 1, "error_estimate": 0.5}


class RungeKutta4:
    """4次のRunge-Kutta法による時間積分"""

    def integrate(
        self,
        dt: float,
        derivative_fn: Callable[[Any, float], Any],
        state: Any,
        **kwargs,
    ) -> Any:
        """4次のRunge-Kutta法で時間積分を実行

        Args:
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        # 各ステージでの微分計算に必要な追加パラメータを保持
        common_kwargs = {**kwargs, "state": state}

        # ステージの微分計算
        k1 = derivative_fn(state, **common_kwargs)
        k2 = derivative_fn(state + 0.5 * dt * k1, **common_kwargs)
        k3 = derivative_fn(state + 0.5 * dt * k2, **common_kwargs)
        k4 = derivative_fn(state + dt * k3, **common_kwargs)

        # 状態の更新
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "method": "Runge-Kutta 4",
            "order": 4,
            "error_estimate": 1 / 30,
        }


def create_integrator(
    integrator_type: str, **kwargs
) -> Union[ForwardEuler, RungeKutta4]:
    """時間積分器を生成

    Args:
        integrator_type: 積分器の種類（"euler", "rk4"）
        **kwargs: 追加のキーワード引数

    Returns:
        生成された時間積分器

    Raises:
        ValueError: 未対応の積分器が指定された場合
    """
    integrators = {
        "euler": ForwardEuler,
        "rk4": RungeKutta4,
    }

    if integrator_type.lower() not in integrators:
        raise ValueError(f"未対応の積分器です: {integrator_type}")

    return integrators[integrator_type.lower()](**kwargs)
