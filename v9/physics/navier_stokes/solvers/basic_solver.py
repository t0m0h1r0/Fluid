"""基本的なNavier-Stokesソルバーを提供するモジュール

このモジュールは、Navier-Stokes方程式の基本的なソルバーを実装します。
時間発展と圧力投影を組み合わせて非圧縮性流れを解きます。
"""

from typing import Dict, Any, List, Optional
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager
from physics.poisson import PoissonSolver, SORSolver
from ..core.base_solver import NavierStokesBase
from ..core.interfaces import NavierStokesTerm, TimeIntegrator
from ..utils.time_integration import RungeKutta4


class BasicNavierStokesSolver(NavierStokesBase):
    """基本的なNavier-Stokesソルバー

    時間発展と圧力投影を組み合わせて非圧縮性流れを解きます。
    """

    def __init__(
        self,
        terms: List[NavierStokesTerm],
        properties: Optional[PropertiesManager] = None,
        time_integrator: Optional[TimeIntegrator] = None,
        poisson_solver: Optional[PoissonSolver] = None,
        logger=None,
    ):
        """ソルバーを初期化

        Args:
            terms: NS方程式の各項
            properties: 物性値マネージャー
            time_integrator: 時間積分スキーム
            poisson_solver: 圧力ポアソンソルバー
            logger: ロガー
        """
        time_integrator = time_integrator or RungeKutta4()
        super().__init__(time_integrator, terms, properties, logger)

        # 圧力ポアソンソルバーの設定
        self.poisson_solver = poisson_solver or SORSolver(
            omega=1.5,
            tolerance=1e-6,
            max_iterations=1000,
            use_redblack=True,
            auto_tune=True,
        )

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
        # 結果を格納するベクトル場
        result = VectorField(velocity.shape, velocity.dx)

        # 各項の寄与を計算して合計
        for term in self.terms:
            if term.enabled:
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
    ) -> ScalarField:
        """圧力ポアソン方程式を解く

        Args:
            velocity: 速度場
            pressure: 現在の圧力場（初期推定値として使用）
            levelset: レベルセット場
            dt: 時間刻み幅

        Returns:
            更新された圧力場
        """
        # 密度場の取得
        density = self.properties.get_density(levelset).data

        # 発散の計算
        div = velocity.divergence()

        # ポアソン方程式の右辺を設定
        rhs = ScalarField(velocity.shape, velocity.dx)
        rhs.data = -div / dt

        # 圧力補正値の計算
        p_corr = ScalarField(velocity.shape, velocity.dx)
        p_corr.data = self.poisson_solver.solve(
            initial_solution=np.zeros_like(pressure.data),
            rhs=rhs.data,
            dx=velocity.dx,
        )

        # 圧力場の更新
        pressure_new = pressure.copy()
        pressure_new.data += p_corr.data

        return pressure_new

    def project_velocity(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        levelset: LevelSetField,
        dt: float,
    ) -> VectorField:
        """速度場を非圧縮に投影

        Args:
            velocity: 速度場
            pressure: 圧力場
            levelset: レベルセット場
            dt: 時間刻み幅

        Returns:
            投影された速度場
        """
        # 密度場の取得
        density = self.properties.get_density(levelset).data

        # 圧力勾配による速度補正
        velocity_new = velocity.copy()
        for i, component in enumerate(velocity_new.components):
            grad_p = np.gradient(pressure.data, velocity.dx, axis=i)
            component.data -= dt * grad_p / density

        return velocity_new

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

            # 2. 圧力ポアソン方程式を解く
            pressure_new = self.solve_pressure_poisson(
                intermediate_state.velocity,
                state.pressure,
                state.levelset,
                dt,
            )

            # 3. 速度場の投影
            velocity_new = self.project_velocity(
                intermediate_state.velocity,
                pressure_new,
                state.levelset,
                dt,
            )

            # 診断情報を収集
            diagnostics = {
                "time": self.time,
                "dt": dt,
                "max_velocity": float(
                    max(np.max(np.abs(comp.data)) for comp in velocity_new.components)
                ),
                "pressure": {
                    "min": float(np.min(pressure_new.data)),
                    "max": float(np.max(pressure_new.data)),
                },
                "poisson_iterations": self.poisson_solver.iteration_count,
            }

            # 各項の診断情報を追加
            for term in self.terms:
                if term.enabled:
                    diagnostics[term.name] = term.get_diagnostics()

            return {
                "velocity": velocity_new,
                "pressure": pressure_new,
                "diagnostics": diagnostics,
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"NavierStokesソルバーの時間発展中にエラー: {e}")
            raise
