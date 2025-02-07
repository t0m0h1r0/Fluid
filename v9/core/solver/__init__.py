"""ソルバーパッケージ

このパッケージは、数値計算のための様々なソルバーの基底クラスを提供します。
"""

from .base import Solver
from .temporal import TemporalSolver
from .iterative import IterativeSolver

__all__ = ["Solver", "TemporalSolver", "IterativeSolver"]
