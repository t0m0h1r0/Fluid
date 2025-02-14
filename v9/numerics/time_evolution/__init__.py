"""
時間発展スキームのパッケージ

数値時間発展計算のための各種積分法を提供します。

主な機能:
- 抽象的な時間積分インターフェース
- 前進オイラー法
- 4次Runge-Kutta法

設計原則:
- SOLID原則に基づく設計
- 汎用的で拡張可能な実装
- 高い型安全性
"""

from .base import TimeIntegrator, TimeIntegratorConfig
from .euler import ForwardEuler
from .runge_kutta import RungeKutta4

__all__ = [
    # 基底クラスとインターフェース
    "TimeIntegrator",
    "TimeIntegratorConfig",
    
    # 具体的な時間積分法
    "ForwardEuler",
    "RungeKutta4",
]

# パッケージのバージョン情報
__version__ = "1.1.0"