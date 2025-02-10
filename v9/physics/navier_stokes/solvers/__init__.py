"""Navier-Stokesソルバーパッケージ

このパッケージは、Navier-Stokes方程式を解くための様々なソルバーを提供します。
基本的なソルバーと、圧力投影法を使用した改良版ソルバーを含みます。
"""

from .basic_solver import BasicNavierStokesSolver
from .projection_solver import ProjectionSolver

__all__ = [
    "BasicNavierStokesSolver",
    "ProjectionSolver",
]
