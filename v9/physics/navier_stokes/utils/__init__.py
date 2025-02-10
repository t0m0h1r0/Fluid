"""Navier-Stokes方程式の解法に関するユーティリティパッケージ

このパッケージは、Navier-Stokes方程式の数値解法に必要な
時間積分スキームや圧力投影法などのユーティリティを提供します。
"""

from .time_integration import (
    ForwardEuler,
    RungeKutta4,
    AdamsBashforth,
    create_integrator,
)
from .projection import (
    ClassicProjection,
    RotationalProjection,
)

__all__ = [
    # 時間積分スキーム
    "ForwardEuler",
    "RungeKutta4",
    "AdamsBashforth",
    "create_integrator",
    # 圧力投影法
    "ClassicProjection",
    "RotationalProjection",
]
