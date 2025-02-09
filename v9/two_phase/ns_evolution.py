from typing import Dict, Any, List, Optional, Type
import numpy as np

from core.field import VectorField, ScalarField
from physics.properties import PropertiesManager
from physics.navier_stokes.terms import (
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
    ForceTerm,
)
from physics.poisson import PoissonSolver, SORSolver
from physics.navier_stokes.timestepping import RungeKutta4, TimeIntegrator

from .base_evolution import BaseEvolution


class NavierStokesEvolution(BaseEvolution):
    """Navier-Stokesの進化を管理するクラス"""

    def __init__(
        self,
        properties_manager: PropertiesManager,
        time_integrator_class: Optional[Type[TimeIntegrator]] = None,
        poisson_solver_class: Optional[Type[PoissonSolver]] = None,
        initial_pressure: Optional[ScalarField] = None,
        terms: Optional[List[Any]] = None,
        use_weno: bool = True,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        *args,
        **kwargs,
    ):
        """Navier-Stokes進化クラスを初期化

        Args:
            properties_manager: 物性値マネージャー
            time_integrator_class: 時間積分クラス（未指定の場合はRK4）
            poisson_solver_class: ポアソンソルバークラス（未指定の場合はSORソルバー）
            initial_pressure: 初期圧力場（オプション）
            terms: カスタムNavier-Stokes項のリスト（オプション）
            use_weno: WENOスキームを使用するかどうか
            max_iterations: 最大反復回数
            tolerance: 収束判定の許容誤差
        """
        super().__init__("NavierStokes", *args, **kwargs)

        # 物性値マネージャー
        self.properties = properties_manager

        # 時間積分器の設定（デフォルトはRK4）
        self.time_integrator_class = time_integrator_class or RungeKutta4
        self.time_integrator = self.time_integrator_class()

        # ポアソンソルバーの設定（デフォルトはSORソルバー）
        self.poisson_solver_class = poisson_solver_class or SORSolver
        self.poisson_solver = self.poisson_solver_class(
            max_iterations=max_iterations, tolerance=tolerance
        )

        # 初期圧力場の設定
        self.initial_pressure = initial_pressure

        # 数値スキームの設定
        default_terms = [
            AdvectionTerm(use_weno=use_weno),
            DiffusionTerm(),
            ForceTerm(),
            PressureTerm(solver=self.poisson_solver, initial_pressure=initial_pressure),
        ]
        self.terms = terms or default_terms

        # 収束条件の設定
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def compute_timestep(
        self, velocity: VectorField, level_set: ScalarField, **kwargs
    ) -> float:
        """時間ステップを計算

        Args:
            velocity: 速度場
            level_set: レベルセット場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間ステップ
        """
        # 各項からの時間ステップ制限を取得
        dt_limits = [
            term.compute_timestep(velocity, **kwargs)
            for term in self.terms
            if hasattr(term, "compute_timestep")
        ]

        return min(dt_limits) * self.cfl if dt_limits else self.max_dt

    def _compute_source_terms(
        self, velocity: VectorField, level_set: ScalarField
    ) -> List[np.ndarray]:
        """各項の寄与を計算

        Args:
            velocity: 速度場
            level_set: レベルセット場

        Returns:
            各方向の速度成分への寄与のリスト
        """
        source_terms = [np.zeros_like(v.data) for v in velocity.components]

        for term in self.terms:
            contributions = term.compute(
                velocity, levelset=level_set, properties=self.properties
            )

            for i, contribution in enumerate(contributions):
                source_terms[i] += contribution

        return source_terms

    def advance(
        self,
        current_velocity: VectorField,
        current_level_set: ScalarField,
        dt: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """速度場を時間発展

        Args:
            current_velocity: 現在の速度場
            current_level_set: 現在のレベルセット場
            dt: 時間ステップ
            **kwargs: 追加のパラメータ

        Returns:
            更新された速度場と診断情報の辞書
        """
        # 各項の寄与を計算
        source_terms = self._compute_source_terms(current_velocity, current_level_set)

        # 時間積分器を使用して速度場を更新
        new_velocity = self.time_integrator.step(current_velocity, dt, source_terms)

        # 診断情報の生成
        diagnostics = {
            "source_terms": {
                term.name: term.get_diagnostics(current_velocity) for term in self.terms
            },
            "max_velocity": max(
                np.max(np.abs(v.data)) for v in new_velocity.components
            ),
        }

        return {"velocity": new_velocity, "diagnostics": diagnostics}
