"""圧力投影法を提供するモジュール

このモジュールは、非圧縮性流体の圧力投影法を実装します。
圧力ポアソン方程式を解いて速度場を非圧縮に投影します。
"""

from typing import Tuple, Optional
import numpy as np
from core.field import VectorField, ScalarField
from physics.poisson import PoissonSolver, SORSolver
from .base import PressureProjection


class ClassicProjection(PressureProjection):
    """古典的な圧力投影法

    Chorinの圧力投影法を実装します。
    圧力勾配による速度場の補正を行います。
    """

    def __init__(
        self,
        poisson_solver: Optional[PoissonSolver] = None,
        rhs_computer=None,
        logger=None,
    ):
        """初期化

        Args:
            poisson_solver: ポアソンソルバー
            rhs_computer: 右辺計算機
            logger: ロガー
        """
        self.poisson_solver = poisson_solver or SORSolver(
            omega=1.5, tolerance=1e-6, max_iterations=100, logger=logger
        )
        self.rhs_computer = rhs_computer
        self.logger = logger

        # 診断情報の初期化
        self._iterations = 0
        self._residuals = []

    def project(
        self, velocity: VectorField, pressure: ScalarField, dt: float, **kwargs
    ) -> Tuple[VectorField, ScalarField]:
        """速度場を非圧縮に投影

        Args:
            velocity: 予測速度場
            pressure: 現在の圧力場
            dt: 時間刻み幅
            **kwargs: 追加のパラメータ

        Returns:
            (補正後の速度場, 更新された圧力場)のタプル
        """
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
            initial_solution=np.zeros_like(pressure.data), rhs=rhs.data, dx=velocity.dx
        )

        # 収束状態の確認
        if not self.poisson_solver.converged:
            if self.logger:
                self.logger.warning(
                    f"圧力ポアソン方程式が収束しませんでした: "
                    f"残差 = {self.poisson_solver.residual_history[-1]:.3e}"
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


class IncPressureProjection(PressureProjection):
    """増分圧力投影法

    圧力の増分を考慮した改良版の投影法です。
    境界条件の扱いが改善されています。
    """

    def __init__(self, **kwargs):
        """初期化"""
        super().__init__(**kwargs)
        # 増分圧力投影法特有の初期化を追加

    def project(
        self, velocity: VectorField, pressure: ScalarField, dt: float, **kwargs
    ) -> Tuple[VectorField, ScalarField]:
        """速度場を非圧縮に投影"""
        # 増分圧力投影法の実装
        # TODO: 実装を追加
        pass
