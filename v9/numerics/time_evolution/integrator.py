from typing import TypeVar
from .base import TimeIntegrator
from .euler import ForwardEuler
from .runge_kutta import RungeKutta4

T = TypeVar("T")

def create_integrator(method: str = "rk4", **kwargs) -> TimeIntegrator[T]:
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