"""Navier-Stokes方程式のソルバーパッケージ

このパッケージは、Navier-Stokes方程式を解くための各種ソルバーを提供します。
"""

from .projection import PressureProjectionSolver

__all__ = [
    "PressureProjectionSolver",
]