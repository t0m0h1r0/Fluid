"""Navier-Stokesソルバーのメインクラスを提供

このモジュールは、非圧縮性Navier-Stokes方程式を解くための
メインのソルバークラスを実装します。
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from core.solver import TemporalSolver
from core.field import VectorField
from .base import NavierStokesBase
from .timestepping import RungeKutta4
from .projection import ClassicProjection
from .terms.advection import AdvectionTerm
from .terms.diffusion import DiffusionTerm
from .terms.force import ForceTerm, GravityForce
from .pressure_rhs import PoissonRHSComputer


class NavierStokesSolver(NavierStokesBase, TemporalSolver):
    """Navier-Stokesソルバークラス

    非圧縮性Navier-Stokes方程式を解くメインのソルバーを実装します。
    時間発展には4次のRunge-Kutta法を、圧力投影には古典的な手法を用います。
    """


def __init__(self, logger=None, use_weno: bool = True, **kwargs):
    """ソルバーを初期化

    Args:
        logger: ロガー
        use_weno: WENOスキームを使用するかどうか
        **kwargs: 基底クラスに渡すパラメータ
    """
    # 物理項の初期化
    force_term = ForceTerm()
    force_term.forces.append(GravityForce())

    terms = [AdvectionTerm(use_weno=use_weno), DiffusionTerm(), force_term]

    # 時間積分とプロジェクションの設定
    time_integrator = RungeKutta4()
    pressure_projection = ClassicProjection(
        rhs_computer=PoissonRHSComputer(),
        logger=logger,  # loggerを渡す
    )

    # 基底クラスの初期化
    NavierStokesBase.__init__(
        self,
        time_integrator=time_integrator,
        pressure_projection=pressure_projection,
        terms=terms,
        logger=logger,  # loggerを渡す
    )
    TemporalSolver.__init__(self, name="NavierStokes", logger=logger, **kwargs)

    def step_forward(
        self, state, dt: Optional[float] = None, **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """1時間ステップを進める

        Args:
            state: 現在の状態
            dt: 時間刻み幅（Noneの場合は自動計算）
            **kwargs: 追加のパラメータ

        Returns:
            (更新された状態, 計算情報を含む辞書)のタプル
        """
        try:
            # 時間刻みの計算
            if dt is None:
                dt = self.compute_timestep(state.velocity, **kwargs)

            # 1. 速度場の時間発展（圧力項を除く）
            velocity_star = self.time_integrator.step(
                state.velocity, dt, self.terms, **kwargs
            )

            # 2. 圧力投影による速度場の補正と圧力場の更新
            velocity_new, pressure_new = self.pressure_projection.project(
                velocity_star,
                state.pressure,
                dt,
                levelset=state.levelset,
                properties=state.properties,
            )

            # 時間と状態の更新
            self._total_time += dt
            self._iteration_count += 1

            new_state = state.copy()
            new_state.velocity = velocity_new
            new_state.pressure = pressure_new
            new_state.time = self._total_time

            # 診断情報の収集
            diagnostics = self._collect_diagnostics(new_state, dt)

            # エラーログの出力を安全に
            if self._logger:
                try:
                    self._logger.info(
                        f"Time: {self._total_time:.3f}, "
                        f"dt = {dt:.3e}, "
                        f"Step: {self._iteration_count}"
                    )
                except Exception as log_error:
                    print(f"ロギング中にエラー: {log_error}")

            return new_state, {
                "dt": dt,
                "time": self._total_time,
                "step": self._iteration_count,
                "diagnostics": diagnostics,
            }

        except Exception as e:
            # エラーハンドリングを安全に
            if self._logger:
                try:
                    self._logger.error(f"ステップ実行中にエラー: {str(e)}")
                except Exception as log_error:
                    print(f"エラーログ出力中にエラー: {log_error}")
            raise

    def solve(self, **kwargs) -> Dict[str, Any]:
        """ソルバーを実行（TemporalSolverの要求するメソッド）

        Returns:
            計算結果と統計情報を含む辞書
        """
        # このメソッドは直接使用されないが、基底クラスの要求により実装
        raise NotImplementedError(
            "NavierStokesSolverでは直接solveを呼び出さず、step_forwardを使用してください"
        )

    def advance(self, dt: float, **kwargs) -> Dict[str, Any]:
        """時間発展を実行（TemporalSolverの要求するメソッド）

        Args:
            dt: 時間刻み幅
            **kwargs: 追加のパラメータ

        Returns:
            計算結果と統計情報を含む辞書
        """
        # このメソッドは直接使用されないが、基底クラスの要求により実装
        raise NotImplementedError(
            "NavierStokesSolverでは直接advanceを呼び出さず、step_forwardを使用してください"
        )

    def initialize(self, **kwargs) -> None:
        """ソルバーを初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        # 各項の初期化
        for term in self.terms:
            if hasattr(term, "initialize"):
                term.initialize(**kwargs)

        # 圧力投影の初期化
        if hasattr(self.pressure_projection, "initialize"):
            self.pressure_projection.initialize(**kwargs)

        # 診断情報のリセット
        self._iteration_count = 0
        self._total_time = 0.0
        self._diagnostics = {}

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算

        Args:
            velocity: 現在の速度場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅
        """
        # 各項からの制限を取得
        dt_limits = []
        for term in self.terms:
            if hasattr(term, "compute_timestep"):
                dt = term.compute_timestep(velocity, **kwargs)
                if dt is not None:
                    dt_limits.append(dt)

        if not dt_limits:
            return self._min_dt

        # 最も厳しい制限を採用
        dt = min(dt_limits)

        # 安全係数を適用
        return self.cfl * dt

    def _collect_diagnostics(self, state, dt: float) -> Dict[str, Any]:
        """診断情報を収集"""
        diag = {}

        # 速度場の情報
        diag["velocity"] = {
            "max": max(np.max(np.abs(v.data)) for v in state.velocity.components),
            "kinetic_energy": 0.5
            * sum(np.sum(v.data**2) for v in state.velocity.components)
            * state.velocity.dx**state.velocity.ndim,
        }

        # 圧力場の情報
        diag["pressure"] = {
            "min": np.min(state.pressure.data),
            "max": np.max(state.pressure.data),
            "l2norm": np.sqrt(np.sum(state.pressure.data**2)),
        }

        # 各項の診断情報
        for term in self.terms:
            diag[term.name] = term.get_diagnostics(state.velocity)

        # 圧力投影の情報
        if hasattr(self.pressure_projection, "_iterations"):
            diag["pressure_projection"] = {
                "iterations": self.pressure_projection._iterations,
                "final_residual": self.pressure_projection._residuals[-1]
                if self.pressure_projection._residuals
                else None,
            }

        return diag
