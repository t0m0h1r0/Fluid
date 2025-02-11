"""Poisson方程式のソルバーパッケージ

このパッケージは、Poisson方程式を解くための各種ソルバーを提供します。
"""

from .base import (
    PoissonSolverBase,
    PoissonSolverTerm,
)
from .config import PoissonSolverConfig as PoissonConfig
from .solver import PoissonSolver
from .methods.sor import SORSolver

__all__ = [
    # ベースクラスとインターフェース
    "PoissonSolverBase",
    "PoissonSolverTerm",
    "PoissonConfig",
    # ソルバー
    "PoissonSolver",
    "SORSolver",
]
