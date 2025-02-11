"""Poisson方程式のソルバーパッケージ

このパッケージは、Poisson方程式を解くための各種ソルバーを提供します。

主な機能:
- 基本的なPoissonソルバーのインターフェース
- SOR法による反復解法
- 共役勾配法（CG）による反復解法
- 設定管理
"""

from .base import (
    PoissonSolverBase,
    PoissonSolverTerm,
)
from .config import PoissonSolverConfig as PoissonConfig
from .solver import PoissonSolver
from .methods.sor import SORSolver
from .methods.cg import ConjugateGradientSolver

__all__ = [
    # ベースクラスとインターフェース
    "PoissonSolverBase",
    "PoissonSolverTerm",
    "PoissonConfig",
    # ソルバー
    "PoissonSolver",
    "SORSolver",
    "ConjugateGradientSolver",
]
