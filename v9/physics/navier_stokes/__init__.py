"""Navier-Stokes方程式関連のパッケージ

このパッケージは、非圧縮性Navier-Stokes方程式の数値計算に関する機能を提供します。
"""

from .solver import NavierStokesSolver
from .projection import ClassicProjection
from .terms import AdvectionTerm, DiffusionTerm, PressureTerm, ForceTerm, GravityForce

__all__ = [
    "NavierStokesSolver",
    "ClassicProjection",
    "AdvectionTerm",
    "DiffusionTerm",
    "PressureTerm",
    "ForceTerm",
    "GravityForce",
]
