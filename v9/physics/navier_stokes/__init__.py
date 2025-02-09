"""Navier-Stokes方程式関連のモジュール

このパッケージは、非圧縮性Navier-Stokes方程式の数値計算に必要な
クラスとユーティリティを提供します。
"""

# ソルバーのインポート
from .solver import NavierStokesSolver

# 各項のインポート
from .terms import AdvectionTerm, DiffusionTerm, PressureTerm, ForceTerm

# 力項のインポート
from .terms.force import GravityForce, SurfaceTensionForce

from .pressure_rhs import PoissonRHSComputer

# エクスポートするモジュール
__all__ = [
    "NavierStokesSolver",
    "AdvectionTerm",
    "DiffusionTerm",
    "PressureTerm",
    "ForceTerm",
    "GravityForce",
    "SurfaceTensionForce",
]
