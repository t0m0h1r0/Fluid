"""Navier-Stokes方程式の各項を提供するパッケージ

このパッケージは、Navier-Stokes方程式を構成する各項（移流項、粘性項、圧力項、外力項）
を実装したクラスを提供します。
"""

from .base import TermBase, ViscousTerm, AdvectiveTerm
from .advection import AdvectionTerm
from .diffusion import DiffusionTerm
from .pressure import PressureTerm
from .force import (
    ForceComponent,
    GravityForce,
    SurfaceTensionForce,
    ForceTerm,
)

__all__ = [
    # 基底クラス
    "TermBase",
    "ViscousTerm",
    "AdvectiveTerm",
    # 具体的な項の実装
    "AdvectionTerm",
    "DiffusionTerm",
    "PressureTerm",
    # 外力関連
    "ForceComponent",
    "GravityForce",
    "SurfaceTensionForce",
    "ForceTerm",
]
