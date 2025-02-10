"""Navier-Stokes方程式の数値計算パッケージ

このパッケージは、Navier-Stokes方程式の数値解法に必要な各種機能を提供します。
主な機能には以下が含まれます：

- ソルバー (NavierStokesSolver, ProjectionSolver)
- 各種項の実装 (移流項、粘性項、圧力項、外力項)
- 時間積分スキーム (RK4, Adams-Bashforth等)
- 圧力投影法 (古典的投影法、回転形式投影法)
"""

# インターフェース
from .core.interfaces import (
    NavierStokesSolver,
    NavierStokesTerm,
    TimeIntegrator,
    PressureProjection,
)

# ソルバー
from .solvers import (
    BasicNavierStokesSolver,
    ProjectionSolver,
)

# 個別の項
from .terms import (
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
    GravityForce,
    SurfaceTensionForce,
    ForceTerm,
)

# ユーティリティ
from .utils import (
    ForwardEuler,
    RungeKutta4,
    AdamsBashforth,
    create_integrator,
    ClassicProjection,
    RotationalProjection,
)

__version__ = "1.0.0"

__all__ = [
    # インターフェース
    "NavierStokesSolver",
    "NavierStokesTerm",
    "TimeIntegrator",
    "PressureProjection",
    # ソルバー
    "BasicNavierStokesSolver",
    "ProjectionSolver",
    # 個別の項
    "AdvectionTerm",
    "DiffusionTerm",
    "PressureTerm",
    "GravityForce",
    "SurfaceTensionForce",
    "ForceTerm",
    # ユーティリティ
    "ForwardEuler",
    "RungeKutta4",
    "AdamsBashforth",
    "create_integrator",
    "ClassicProjection",
    "RotationalProjection",
]
