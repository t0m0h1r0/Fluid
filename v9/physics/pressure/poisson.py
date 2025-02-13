"""
二相流体における圧力ポアソン方程式（PPE）のソルバー

圧力ポアソン方程式:
∇²p = ∇⋅(-ρ Du/Dt + ∇⋅τ + f)

簡略化形:
∇²p ≈ ρ(∇⋅(u⋅∇u) + ∇⋅fs - ∇⋅(∇ν⋅∇u))
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from core.field import VectorField, ScalarField
from numerics.poisson import PoissonConfig, ConjugateGradientSolver
from .terms import AdvectionTerm, ViscousTerm, ForceTerm


class PressurePoissonSolver:
    """圧力ポアソン方程式のソルバー"""

    def __init__(self, solver_config: Optional[PoissonConfig] = None):
        """
        Args:
            solver_config: ポアソンソルバーの設定
        """
        # 基本ソルバーの初期化
        self._poisson_solver = ConjugateGradientSolver(solver_config or PoissonConfig())

        # 各項の初期化
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
        """
        圧力ポアソン方程式を解く

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            external_force: 外力場（オプション）

        Returns:
            (圧力場, 診断情報)のタプル
        """
        # 1. 各項の計算
        # 移流項: ∇⋅(u⋅∇u)
        advection_term = self._advection_term.compute(velocity=velocity)

        # 粘性項: -∇⋅(∇ν⋅∇u)
        viscous_term = self._viscous_term.compute(
            velocity=velocity, viscosity=viscosity
        )

        # 外力項: ∇⋅f
        force_term = self._force_term.compute(
            shape=velocity.shape, dx=velocity.dx, external_force=external_force
        )

        # 2. 右辺の組み立て
        rhs = ScalarField(velocity.shape, velocity.dx)
        rhs.data = density.data * (
            advection_term.data + viscous_term.data + force_term.data
        )

        # 3. ポアソン方程式を解く
        pressure = ScalarField(velocity.shape, velocity.dx)
        pressure.data = self._poisson_solver.solve(rhs.data)

        # 診断情報の更新
        self._update_diagnostics(
            pressure=pressure,
            rhs=rhs,
            terms={
                "advection": self._advection_term.get_diagnostics(),
                "viscous": self._viscous_term.get_diagnostics(),
                "force": self._force_term.get_diagnostics(),
            },
        )

        return pressure, self._diagnostics

    def _update_diagnostics(
        self, pressure: ScalarField, rhs: ScalarField, terms: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        診断情報を更新

        Args:
            pressure: 計算された圧力場
            rhs: 右辺
            terms: 各項の診断情報
        """
        self._diagnostics = {
            "pressure": {
                "min": float(np.min(pressure.data)),
                "max": float(np.max(pressure.data)),
                "mean": float(np.mean(pressure.data)),
                "norm": float(np.linalg.norm(pressure.data)),
            },
            "source_terms": terms,
            "rhs": {
                "min": float(np.min(rhs.data)),
                "max": float(np.max(rhs.data)),
                "norm": float(np.linalg.norm(rhs.data)),
            },
        }

        # ポアソンソルバーの診断情報も追加
        if hasattr(self._poisson_solver, "get_diagnostics"):
            self._diagnostics["solver"] = self._poisson_solver.get_diagnostics()
