"""Navier-Stokes方程式の各項を提供するパッケージ

このパッケージは、Navier-Stokes方程式を構成する各項
（移流項、粘性項、圧力項、加速度項）を実装したクラスを提供します。
"""

from .solver import NavierStokesSolver

__all__ = ["NavierStokesSolver"]
