"""Navier-Stokes方程式のソルバーを提供するモジュール

このモジュールは、非圧縮性Navier-Stokes方程式を解くためのソルバークラスを提供します。
時間発展は新しい時間発展フレームワークに基づいて実装されます。
"""

from typing import Dict, Any, List, Optional
import numpy as np

from core.field import VectorField
from core.time_evolution import TimeEvolutionSolver
from physics.properties import PropertiesManager
from physics.poisson import PoissonSolver, SORSolver
from .terms import AdvectionTerm, DiffusionTerm, ForceTerm, GravityForce
from .projection import ClassicProjection


class NavierStokesSolver(TimeEvolutionSolver):
    """Navier-Stokesソルバークラス"""

    def __init__(
        self,
        use_weno: bool = True,
        properties: Optional[PropertiesManager] = None,
        poisson_solver: Optional[PoissonSolver] = None,
        logger=None,
    ):
        """ソルバーを初期化

        Args:
            use_weno: WENOスキームを使用するかどうか
            properties: 物性値マネージャー
            poisson_solver: 圧力ポアソンソルバー
            logger: ロガー
        """
        super().__init__(logger=logger)
        self.use_weno = use_weno
        self.properties = properties

        # 圧力投影法の設定
        self.poisson_solver = poisson_solver or SORSolver(
            omega=1.5,
            tolerance=1e-6,
            max_iterations=1000,
            use_redblack=True,
            auto_tune=True,
        )
        self.pressure_projection = ClassicProjection(self.poisson_solver)

        # 各項の初期化
        self.terms = self._initialize_terms()

    def _initialize_terms(self) -> List[Any]:
        """NavierStokes方程式の各項を初期化"""
        # 移流項
        advection = AdvectionTerm(use_weno=self.use_weno)

        # 粘性項
        diffusion = DiffusionTerm()

        # 外力項（重力）
        force = ForceTerm()
        force.forces.append(GravityForce())

        return [advection, diffusion, force]

    def compute_timestep(self, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算

        Args:
            **kwargs: 必要なパラメータ
                - state: 現在のシミュレーション状態

        Returns:
            計算された時間刻み幅
        """
        state = kwargs.get("state")
        if state is None:
            raise ValueError("stateが指定されていません")

        # 速度の最大値を計算
        velocity = state.velocity
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # 粘性による制限
        nu = self.properties.get_kinematic_viscosity(state.levelset)
        nu_max = np.max(nu)
        dx = velocity.dx

        # 移流と粘性による制限のうち厳しい方を採用
        dt_adv = self.cfl * dx / (max_velocity + 1e-10)
        dt_vis = 0.5 * dx**2 / (nu_max + 1e-10)

        return min(dt_adv, dt_vis)

    def compute_derivative(self, state: Any, **kwargs) -> VectorField:
        """速度場の時間微分を計算

        Args:
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間微分
        """
        # 各項の寄与を計算
        contributions = []
        for term in self.terms:
            contrib = term.compute(
                state.velocity, state.levelset, state.properties, **kwargs
            )
            contributions.append(contrib)

        # 結果を格納するベクトル場
        result = VectorField(state.velocity.shape, state.velocity.dx)

        # 各項の寄与を足し合わせる
        for i in range(len(result.components)):
            result.components[i].data = sum(c[i] for c in contributions)

        return result

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
            result = super().step_forward(dt, **kwargs)
            velocity_star = result["state"]

            # 2. 圧力投影による速度場の補正と圧力場の更新
            velocity_new, pressure_new = self.pressure_projection.project(
                velocity_star,
                state.pressure,
                dt,
                levelset=state.levelset,
                properties=self.properties,
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
                diagnostics[term.name] = term.get_diagnostics(
                    velocity_new, state.levelset, state.properties
                )

            return {
                "velocity": velocity_new,
                "pressure": pressure_new,
                "diagnostics": diagnostics,
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"NavierStokesソルバーの時間発展中にエラー: {e}")
            raise
