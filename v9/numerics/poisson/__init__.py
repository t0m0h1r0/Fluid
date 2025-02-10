"""Poisson方程式のソルバーパッケージ

このパッケージは、Poisson方程式を解くための各種ソルバーを提供します。
"""

from .base import (
    PoissonSolverBase,
    PoissonSolverConfig as PoissonConfig,
    PoissonSolverTerm,
)
from .solver import PoissonSolver
from .methods.sor import SORSolver

__all__ = [
    # ベースクラスとインターフェース
    "PoissonSolverBase",
    "PoissonConfig",
    "PoissonSolverTerm",
    # ソルバー
    "PoissonSolver",
    "SORSolver",
]
