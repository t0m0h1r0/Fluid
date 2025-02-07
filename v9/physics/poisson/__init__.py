"""Poisson方程式のソルバーを提供するモジュール

このパッケージは、Poisson方程式を解くための各種ソルバーを提供します。
"""

# ソルバーのインポート
from .solver import PoissonSolver
from .sor import SORSolver

# エクスポートするモジュール
__all__ = ["PoissonSolver", "SORSolver"]
