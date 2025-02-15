"""多相流体計算パッケージ

このパッケージは、多相流体計算に必要な機能を提供します。
Level Set法による界面追跡やVOF法による界面捕獲など、
様々な手法に対応可能な汎用的なインターフェースを提供します。
"""

from .interface import InterfaceOperations

__all__ = ["InterfaceOperations"]

# パッケージのバージョン情報
__version__ = "1.0.0"
