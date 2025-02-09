"""圧力投影法を提供するモジュール"""

from typing import Tuple, Optional
import numpy as np
from core.field import VectorField, ScalarField
from physics.poisson import PoissonSolver, SORSolver
from .base import NSComponentBase


class ClassicProjection(NSComponentBase):
    """古典的な圧力投影法"""

    def __init__(
        self, poisson_solver: Optional[PoissonSolver] = None, rhs_computer=None
    ):
        """初期化

        Args:
            poisson_solver: ポアソンソルバー
            rhs_computer: 右辺計算機
        """
        super().__init__(None)  # loggerはNavierStokesBaseから継承

        self.poisson_solver = poisson_solver or SORSolver(
            omega=1.5,  # より適切な緩和係数
            tolerance=1e-8,  # より厳しい収束判定
            max_iterations=1000,  # 反復回数を増やす
            use_redblack=True,  # より効率的な反復
            auto_tune=True,  # 自動調整を有効化
        )
        self.rhs_computer = rhs_computer

        # 診断情報の初期化
        self._iterations = 0
        self._residuals = []

    def project(
        self, velocity: VectorField, pressure: ScalarField, dt: float, **kwargs
    ) -> Tuple[VectorField, ScalarField]:
        """速度場を非圧縮に投影"""
        try:
            # 右辺の計算
            if self.rhs_computer:
                rhs = self.rhs_computer.compute(velocity, **kwargs)
            else:
                # 単純な発散に基づく右辺
                rhs = ScalarField(velocity.shape, velocity.dx)
                rhs.data = velocity.divergence().data / dt

            # 圧力補正値の計算
            p_corr = ScalarField(pressure.shape, pressure.dx)
            p_corr.data = self.poisson_solver.solve(
                initial_solution=np.zeros_like(pressure.data),
                rhs=rhs.data,
                dx=velocity.dx,
            )

            # 収束状態の確認
            if not self.poisson_solver.converged:
                self.log(
                    "warning",
                    f"圧力ポアソン方程式が収束しませんでした: "
                    f"残差 = {self.poisson_solver.residual_history[-1]:.3e}",
                )

            # 診断情報の更新
            self._iterations = self.poisson_solver.iteration_count
            self._residuals = self.poisson_solver.residual_history

            # 速度場の補正
            result_velocity = velocity.copy()
            for i in range(velocity.ndim):
                grad_p = np.gradient(p_corr.data, velocity.dx, axis=i)
                result_velocity.components[i].data -= dt * grad_p

            # 圧力場の更新
            result_pressure = pressure.copy()
            result_pressure.data += p_corr.data

            return result_velocity, result_pressure

        except Exception as e:
            self.log("error", f"圧力投影中にエラー: {e}")
            raise
