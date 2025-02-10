"""時間発展計算パッケージ

このパッケージは、時間発展計算に必要な基底クラスとユーティリティを提供します。
"""

from .base import TimeEvolutionBase, TimeEvolutionTerm, TimeEvolutionConfig
from .solver import TimeEvolutionSolver
from .integrator import TimeIntegratorBase, create_integrator
from .methods.euler import ForwardEuler
from .methods.runge_kutta import RungeKutta4

__all__ = [
    # 基底クラスとインターフェース
    "TimeEvolutionBase",
    "TimeEvolutionTerm",
    "TimeEvolutionConfig",
    "TimeIntegratorBase",
    # ソルバー
    "TimeEvolutionSolver",
    # 積分器関連
    "create_integrator",
    "ForwardEuler",
    "RungeKutta4",
]
