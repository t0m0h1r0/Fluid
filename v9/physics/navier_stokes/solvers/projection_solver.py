"""圧力投影法を使用したNavier-Stokesソルバーを提供するモジュール

このモジュールは、圧力投影法を使用した改良版のNavier-Stokesソルバーを実装します。
より高精度な圧力場の計算と厳密な非圧縮性の保証を提供します。
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.levelset.properties import PropertiesManager
from physics.poisson import PoissonSolver, SORSolver
from ..core.base_solver import NavierStokesBase
from ..core.interfaces import NavierStokesTerm, TimeIntegrator
from ..utils.time_integration import RungeKutta4
from ..utils.projection import ClassicProjection, RotationalProjection


class ProjectionSolver(NavierStokesBase):
    """圧力投影法を用いたNavier-Stokesソルバー

    圧力投影法を使用して非圧縮性を厳密に満たすソルバーです。
    古典的な投影法と回転形式の投影法をサポートします。
    """

    def __init__(
        self,
        terms: List[NavierStokesTerm],
        properties: Optional[PropertiesManager] = None,
        time_integrator: Optional[TimeIntegrator] = None,
        poisson_solver: Optional[PoissonSolver] = None,
        use_rotational: bool = False,
        logger=None,
    ):
        """ソルバーを初期化

        Args:
            terms: NS方程式の各項
            properties: 物性値マネージャー
            time_integrator: 時間積分スキーム
            poisson_solver: 圧力ポアソンソルバー
            use_rotational: 回転形式の投影法を使用するかどうか
            logger: ロガー
        """
        time_integrator = time_integrator or RungeKutta4()
        super().__init__(time_integrator, terms, properties, logger)

        # 圧力投影法の設定
        self.poisson_solver = poisson_solver or SORSolver(
            omega=1.5,
            tolerance=1e-6,
            max_iterations=1000,
            use_redblack=True,
            auto_tune=True,
        )

        # 投影法の選択
        self.projection = (
            RotationalProjection(self.poisson_solver)
            if use_rotational
            else ClassicProjection(self.poisson_solver)
        )

        # 診断情報の初期化
        self._projection_diagnostics: Dict[str, Any] = {}

    def compute_derivative(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> VectorField:
        """速度場の時間微分を計算

        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間微分のVectorField
        """
        # 圧力項を除く全項の寄与を計算
        result = VectorField(velocity.shape, velocity.dx)

        # 各項の寄与を計算して合計（圧力項を除く）
        for term in self.terms:
            if term.enabled and term.name != "Pressure":
                contributions = term.compute(velocity, levelset, properties, **kwargs)
                # 各成分に寄与を加算
                for i, contribution in enumerate(contributions):
                    result.components[i].data += contribution

        return result

    def solve_pressure_poisson(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        levelset: LevelSetField,
        dt: float,
    ) -> Tuple[VectorField, ScalarField]:
        """圧力ポアソン方程式を解き、速度場を投影

        Args:
            velocity: 速度場
            pressure: 現在の圧力場
            levelset: レベルセット場
            dt: 時間刻み幅

        Returns:
            (投影された速度場, 更新された圧力場)のタプル
        """
        return self.projection.project(
            velocity=velocity,
            pressure=pressure,
            dt=dt,
            levelset=levelset,
            properties=self.properties,
        )

    def step_forward(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める

        Args:
            dt: 時間刻み幅
            **kwargs: 追加のパラメータ
                - state: 現在のシミュレーション状態

        Returns:
            計算結果と診断情報を含む辞書
        """
        state = kwargs.get("state")
        if state is None:
            raise ValueError("stateが指定されていません")

        try:
            # 1. 速度場の時間発展（圧力項を除く）
            intermediate_state = super().step_forward(dt=dt, state=state)["state"]

            # 2. 圧力投影による速度場の補正と圧力場の更新
            velocity_new, pressure_new = self.solve_pressure_poisson(
                intermediate_state.velocity,
                state.pressure,
                state.levelset,
                dt,
            )

            # 診断情報を収集
            diagnostics = self._collect_diagnostics(
                velocity_new, pressure_new, intermediate_state, dt
            )

            return {
                "velocity": velocity_new,
                "pressure": pressure_new,
                "diagnostics": diagnostics,
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"ProjectionSolverの時間発展中にエラー: {e}")
            raise

    def _collect_diagnostics(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        intermediate_state: Any,
        dt: float,
    ) -> Dict[str, Any]:
        """診断情報を収集

        Args:
            velocity: 更新後の速度場
            pressure: 更新後の圧力場
            intermediate_state: 中間状態
            dt: 時間刻み幅

        Returns:
            診断情報を含む辞書
        """
        # 基本的な診断情報
        diag = {
            "time": self.time,
            "dt": dt,
            "iteration": self.iteration_count,
            "velocity": {
                "max": float(max(np.max(np.abs(c.data)) for c in velocity.components)),
                "divergence": float(np.max(np.abs(velocity.divergence()))),
                "energy": float(
                    sum(np.sum(c.data**2) for c in velocity.components)
                    * 0.5
                    * velocity.dx**3
                ),
            },
            "pressure": {
                "min": float(np.min(pressure.data)),
                "max": float(np.max(pressure.data)),
                "mean": float(np.mean(pressure.data)),
            },
            "projection": {
                "type": "rotational"
                if isinstance(self.projection, RotationalProjection)
                else "classic",
                "poisson_iterations": self.poisson_solver.iteration_count,
            },
        }

        # 各項の診断情報を追加
        for term in self.terms:
            if term.enabled:
                diag[term.name] = term.get_diagnostics()

        return diag

    def get_diagnostics(self) -> Dict[str, Any]:
        """ソルバーの診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "projection_type": "rotational"
                if isinstance(self.projection, RotationalProjection)
                else "classic",
                "terms": {
                    term.name: term.get_diagnostics()
                    for term in self.terms
                    if term.enabled
                },
                "poisson_solver": {
                    "iterations": self.poisson_solver.iteration_count,
                    "converged": getattr(self.poisson_solver, "converged", None),
                },
            }
        )
        return diag
