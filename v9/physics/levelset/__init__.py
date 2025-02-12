"""Level Set法のパッケージ

このパッケージは、Level Set法に関する機能を提供します。

主な機能:
- Level Set関数のフィールド管理
- 幾何学的計算（法線ベクトル、曲率など）
- Heaviside関数とDelta関数による密度場の計算
- 界面の初期化と再初期化
"""

from .field import LevelSetField, LevelSetParameters

__all__ = [
    "LevelSetField",
    "LevelSetParameters",
]

# バージョン情報
__version__ = "1.0.0"
