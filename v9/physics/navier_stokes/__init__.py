"""Navier-Stokes方程式の数値計算パッケージ

このパッケージは、Navier-Stokes方程式の数値解法に必要な各種機能を提供します。
主な機能には以下が含まれます：

- 項の実装 (移流項、粘性項、圧力項、外力項)
- 圧力投影法ソルバー
"""

from .terms.advection import AdvectionTerm
from .terms.diffusion import DiffusionTerm
from .terms.force import GravityForce, SurfaceTensionForce
from .terms.pressure import PressureTerm
from .solvers.projection import PressureProjectionSolver

__all__ = [
    # 項の実装
    "AdvectionTerm",
    "DiffusionTerm",
    "PressureTerm",
    "GravityForce",
    "SurfaceTensionForce",
    # ソルバー
    "PressureProjectionSolver",
]
