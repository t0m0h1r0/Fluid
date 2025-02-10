"""Navier-Stokes方程式のコアインターフェースを提供するパッケージ

このパッケージは、Navier-Stokes方程式に関する基本的なプロトコルとインターフェースを定義します。
"""

from .interfaces import (
    NavierStokesTerm,
    NavierStokesSolver
)

__all__ = [
    "NavierStokesTerm",
    "NavierStokesSolver",
]