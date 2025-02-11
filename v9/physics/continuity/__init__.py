"""
連続の方程式モジュールのパッケージ初期化

このパッケージは、流体力学における連続の方程式を実装します。
"""

from .continuity import ContinuityEquation

__all__ = [
    "ContinuityEquation",
]
