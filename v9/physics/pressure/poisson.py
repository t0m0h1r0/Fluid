"""
圧力ポアソン方程式のソルバー

圧力ポアソン方程式: ∇²p = ∇⋅f を解きます。

ここで、右辺の f は以下の項から構成されます：
- 移流項: -ρ(u⋅∇)u
- 粘性項: μ∇²u
- 外力項: ρg + f_s (重力と表面張力)
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

from core.field import VectorField, ScalarField
from numerics.poisson import PoissonConfig, ConjugateGradientSolver
from .terms import AdvectionTerm, ViscousTerm, ForceTerm


class PressurePoissonSolver:
    """圧力ポアソン方程式のソルバー

    非圧縮性流体の圧力場を計算するためのソルバーです。
    各種の物理項（移流、粘性、外力）の発散から右辺を構築し、
    ポアソン方程式を解いて圧力場を得ます。
    """

    def __init__(self, solver_config: Optional[PoissonConfig] = None):
        """
        Args:
            solver_config: ポアソンソルバーの設定
        """
        # ポアソンソルバーの初期化
        self._poisson_solver = ConjugateGradientSolver(solver_config or PoissonConfig())

        # 物理項の初期化
        self._advection_term = AdvectionTerm()
        self._viscous_term = ViscousTerm()
        self._force_term = ForceTerm()

        # 診断情報の初期化
        self._diagnostics: Dict[str, Any] = {}

    def solve(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        external_force: Optional[VectorField] = None,
        **kwargs,
    ) -> Tuple[ScalarField, Dict[str, Any]]:
        """圧力ポアソン方程式を解く

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            external_force: 外力場（オプション）
            **kwargs: 追加のパラメータ

        Returns:
            (圧力場, 診断情報)のタプル
        """
        # 各物理項からの寄与を計算
        source_terms = self._compute_source_terms(
            velocity, density, viscosity, external_force
        )

        # 右辺の構築
        rhs = ScalarField(velocity.shape, velocity.dx)
        for term in source_terms.values():
            rhs.data = np.array(rhs.data) + np.array(term.data)

        # ポアソン方程式を解く
        pressure = ScalarField(velocity.shape, velocity.dx)
        pressure.data = self._poisson_solver.solve(np.array(rhs.data))

        # 診断情報の更新
        self._update_diagnostics(pressure, rhs, source_terms)

        return pressure, self._diagnostics

    def _compute_source_terms(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        external_force: Optional[VectorField],
    ) -> Dict[str, ScalarField]:
        """ポアソン方程式の右辺を構成する各項を計算"""
        source_terms = {}

        # 移流項の発散: -∇⋅(ρ(u⋅∇)u)
        advection = self._advection_term.compute(velocity=velocity)
        source_terms["advection"] = ScalarField(
            velocity.shape,
            velocity.dx,
        )

        # 粘性項の発散: ∇⋅(μ∇²u)
        viscous = self._viscous_term.compute(velocity=velocity, viscosity=viscosity)
        source_terms["viscous"] = ScalarField(
            velocity.shape, velocity.dx,
        )

        # 外力項の発散: ∇⋅f
        if external_force is not None:
            force = self._force_term.compute(
                shape=velocity.shape, dx=velocity.dx, external_force=external_force
            )
            source_terms["force"] = force

        return source_terms

    def _update_diagnostics(
        self,
        pressure: ScalarField,
        rhs: ScalarField,
        source_terms: Dict[str, ScalarField],
    ) -> None:
        """診断情報を更新

        Args:
            pressure: 計算された圧力場
            rhs: 右辺
            source_terms: 各物理項からの寄与
        """
        self._diagnostics = {
            "pressure": {
                "min": float(np.min(pressure.data)),
                "max": float(np.max(pressure.data)),
                "mean": float(np.mean(pressure.data)),
                "norm": float(np.linalg.norm(pressure.data)),
            },
            "source_terms": {
                name: {
                    "min": float(np.min(term.data)),
                    "max": float(np.max(term.data)),
                    "norm": float(np.linalg.norm(term.data)),
                }
                for name, term in source_terms.items()
            },
            "rhs": {
                "min": float(np.min(rhs.data)),
                "max": float(np.max(rhs.data)),
                "norm": float(np.linalg.norm(rhs.data)),
            },
        }

        # ポアソンソルバーの診断情報も追加
        if hasattr(self._poisson_solver, "get_diagnostics"):
            self._diagnostics["solver"] = self._poisson_solver.get_diagnostics()
