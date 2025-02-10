"""Navier-Stokes方程式ソルバーの実装"""

from typing import Dict, Any, Optional
import numpy as np

from physics.navier_stokes import (
    NavierStokesSolver,
    ProjectionSolver,
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
    GravityForce,
    SurfaceTensionForce,
    ForceTerm,
    ClassicProjection,
)
from physics.levelset.properties import PropertiesManager
from ..state import SimulationState


class ProjectionNavierStokesSolver(NavierStokesSolver):
    """投影法を用いたNavier-Stokes方程式ソルバー"""

    def __init__(
        self,
        use_weno: bool = True,
        properties: Optional[PropertiesManager] = None,
        logger=None,
    ):
        """ソルバーを初期化"""
        super().__init__()
        self.logger = logger
        self.properties = properties

        # 基本の項を設定
        self.terms = [AdvectionTerm(use_weno=use_weno), DiffusionTerm(), PressureTerm()]

        # 外力項を設定
        force_term = ForceTerm()
        force_term.add_force(GravityForce())
        if properties is not None:
            force_term.add_force(
                SurfaceTensionForce(properties.get_surface_tension_coefficient() or 0.0)
            )
        self.terms.append(force_term)

        # 圧力投影法を設定
        self.projection = ClassicProjection(ProjectionSolver())

    def initialize(self, state: SimulationState):
        """初期化処理"""
        if self.logger:
            self.logger.info("Navier-Stokesソルバーを初期化")

    def compute_timestep(self, state: SimulationState, **kwargs) -> float:
        """時間刻み幅を計算"""
        dt_limits = []

        # 各項からの時間刻み幅制限を計算
        for term in self.terms:
            try:
                dt = term.compute_timestep(
                    velocity=state.velocity,
                    levelset=state.levelset,
                    properties=self.properties,
                    pressure=state.pressure,
                )
                dt_limits.append(dt)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"項{term.name}の時間刻み幅計算でエラー: {e}")

        return min(dt_limits) if dt_limits else 1.0

    def step_forward(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """1時間ステップを進める"""
        try:
            # 各項の寄与を計算
            derivatives = []
            for term in self.terms:
                try:
                    derivative = term.compute(
                        velocity=state.velocity,
                        levelset=state.levelset,
                        properties=self.properties,
                        pressure=state.pressure,
                    )
                    derivatives.append(derivative)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"項{term.name}の計算中にエラー: {e}")
                    raise

            # 時間発展を計算
            velocity_star = state.velocity.copy()
            for i, component in enumerate(velocity_star.components):
                component.data += dt * sum(d[i] for d in derivatives)

            # 圧力投影を実行
            velocity_new, pressure_new = self.projection.project(
                velocity=velocity_star,
                pressure=state.pressure,
                dt=dt,
                levelset=state.levelset,
                properties=self.properties,
            )

            # 診断情報の収集
            diagnostics = {
                "max_velocity": float(
                    max(np.max(np.abs(c.data)) for c in velocity_new.components)
                ),
                "pressure": {
                    "min": float(np.min(pressure_new.data)),
                    "max": float(np.max(pressure_new.data)),
                },
            }

            return {
                "velocity": velocity_new,
                "pressure": pressure_new,
                "diagnostics": diagnostics,
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"時間発展計算中にエラー: {e}")
            raise

    def finalize(self):
        """終了処理"""
        if self.logger:
            self.logger.info("Navier-Stokesソルバーを終了")
