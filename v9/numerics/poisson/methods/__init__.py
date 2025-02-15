"""
Poisson方程式ソルバーパッケージ

このパッケージは、偏微分方程式における
Poisson方程式の高速・高精度な数値解法を提供します。

主な特徴:
- JAXベースの実装
- 複数の数値解法
- 高度な前処理と収束加速
"""

from .base import PoissonSolverBase
from .sor import PoissonSORSolver
from .multigrid import PoissonMultigridSolver
from .cg import PoissonCGSolver

__all__ = [
    # 基底クラスとインターフェース
    "PoissonSolverBase",
    "PoissonSolverConfig",
    "PoissonConfig",
    # 具体的なソルバー実装
    "PoissonCGSolver",
    "PoissonMultigridSolver",
    "PoissonSORSolver",
]

# パッケージバージョン
__version__ = "1.0.2"