"""圧力ポアソン方程式のソルバーパッケージ

このパッケージは、二相流体における圧力ポアソン方程式を解くためのソルバーを提供します。
主な機能：
- 密度が空間的に変化する場合の圧力場の計算
- 非圧縮性条件の維持
- 界面張力などの外力の考慮
"""

from .poisson import PressurePoissonSolver

__all__ = [
    "PressurePoissonSolver",
]