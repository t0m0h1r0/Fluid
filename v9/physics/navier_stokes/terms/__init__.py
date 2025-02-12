"""Navier-Stokes方程式の各項を提供するパッケージ

このパッケージは、Navier-Stokes方程式を構成する各項
（移流項、粘性項、圧力項）を実装したクラスを提供します。
"""

from .base import BaseNavierStokesTerm
from .advection import AdvectionTerm
from .diffusion import DiffusionTerm
from .pressure import PressureTerm

__all__ = [
    # 基底クラス
    "BaseNavierStokesTerm",
    # 具体的な項の実装
    "AdvectionTerm",
    "DiffusionTerm",
    "PressureTerm",
]