"""Navier-Stokesソルバーのメインクラスを提供するモジュール

このモジュールは、非圧縮性Navier-Stokes方程式を解くためのソルバーを実装します。
分離解法（Projection Method）を用いて、各時間ステップで予測子-修正子法を適用します。
"""

from typing import Dict, Any, Optional
import numpy as np
import traceback

from core.solver import TemporalSolver
from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.poisson.solver import PoissonSolver
from physics.poisson.sor import SORSolver
from .terms.advection import AdvectionTerm
from .terms.diffusion import DiffusionTerm
from .terms.pressure import PressureTerm
from .terms.force import ForceTerm, GravityForce, SurfaceTensionForce


class NavierStokesSolver(TemporalSolver):
    """Navier-Stokesソルバークラス

    非圧縮性Navier-Stokes方程式を解くソルバーを実装します。
    分離解法を用いて、圧力と速度を分離して解きます。
    """

    def __init__(
        self,
        logger=None,
        use_weno: bool = True,
        poisson_solver: Optional[PoissonSolver] = None,
        **kwargs
    ):
        """Navier-Stokesソルバーを初期化

        Args:
            logger: ロガー
            use_weno: WENOスキームを使用するかどうか
            poisson_solver: 圧力Poisson方程式のソルバー
            **kwargs: 基底クラスに渡すパラメータ
        """
        # 基底クラスの初期化
        super().__init__(name="NavierStokes", **kwargs)

        # ロガーの設定
        self.logger = logger

        # 各項の初期化
        self.advection = AdvectionTerm(use_weno=use_weno)
        self.diffusion = DiffusionTerm()
        self.pressure = PressureTerm()
        self.force = ForceTerm()

        # デフォルトの外力として重力を追加
        self.force.add_force(GravityForce())

        # Poissonソルバーの設定
        self.poisson_solver = poisson_solver or SORSolver(
            omega=1.5,
            tolerance=1e-6,
            max_iterations=100
        )

        # 診断用の変数
        self._max_divergence = 0.0
        self._pressure_iterations = 0

    def solve(self, **kwargs) -> Dict[str, Any]:
        """ソルバーを実行（具体的な実装は他のメソッドで行う）

        他のメソッドで実装されるため、基本的には例外を投げる。
        """
        raise NotImplementedError("時間発展ソルバーの具体的な実装は`advance`メソッドで行います。")

    def initialize(self, state=None, **kwargs):
        """ソルバーの初期化

        Args:
            state: 初期状態（オプション）
            **kwargs: 追加のパラメータ
        """
        # 表面張力の初期化（二相流体の場合）
        levelset = kwargs.get('levelset')
        if levelset is not None and not any(
            isinstance(f, SurfaceTensionForce) for f in self.force.forces
        ):
            self.force.add_force(SurfaceTensionForce())

        # 各項の初期化
        self.advection.initialize(**kwargs)
        self.diffusion.initialize(**kwargs)
        self.pressure.initialize(**kwargs)
        self.force.initialize(**kwargs)

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算

        Args:
            velocity: 現在の速度場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅
        """
        # 各項からの制限を取得
        dt_advection = self.advection.compute_timestep(velocity, **kwargs)
        dt_diffusion = self.diffusion.compute_timestep(velocity, **kwargs)

        # 最も厳しい制限を採用
        dt = min(dt_advection, dt_diffusion)

        # 安全係数を適用
        return self.cfl * dt

    def advance(
        self,
        state,
        dt: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """1時間ステップを進める

        分離解法を用いて、以下のステップで解きます：
        1. 予測子ステップ：圧力項を除いた方程式を解く
        2. 圧力補正ステップ：圧力Poisson方程式を解く
        3. 修正子ステップ：速度場を補正

        Args:
            state: 現在の状態
            dt: 時間刻み幅（Noneの場合は自動計算）
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態と計算情報
        """
        try:
            # 時間刻みの計算
            if dt is None:
                dt = self.compute_timestep(state.velocity, **kwargs)

            # 圧力場と速度場を取得
            velocity = state.velocity
            pressure = state.pressure
            levelset = getattr(state, "levelset", None)

            # 予測子ステップ
            velocity_star = self._predictor_step(dt, velocity, levelset, **kwargs)

            # 圧力補正ステップ
            pressure_correction = self._pressure_correction_step(
                dt, velocity_star, pressure, **kwargs
            )

            # 修正子ステップ
            velocity_new = self._corrector_step(
                dt, velocity_star, pressure_correction, **kwargs
            )

            # 非圧縮性条件のチェック
            div = velocity_new.divergence()
            self._max_divergence = np.max(np.abs(div.data))

            # 圧力場の更新
            pressure.data += pressure_correction.data

            # 状態の更新
            state.velocity = velocity_new
            state.pressure = pressure

            # 診断情報の収集
            diagnostics = self._collect_diagnostics(
                velocity_new, pressure, levelset, div, **kwargs
            )

            return state, diagnostics

        except Exception as e:
            # エラーログの出力
            if self.logger:
                self.logger.error(f"シミュレーション実行中にエラーが発生: {e}")
                self.logger.error(traceback.format_exc())

            # エラー時の状態を返す（または例外を再送出）
            raise

    def _predictor_step(
        self,
        dt: float,
        velocity: VectorField,
        levelset: Optional[LevelSetField] = None,
        **kwargs,
    ) -> VectorField:
        """予測子ステップ

        圧力項を除いた方程式を解きます。

        Args:
            dt: 時間刻み幅
            velocity: 現在の速度場
            levelset: Level Set場
            **kwargs: 追加のパラメータ

        Returns:
            予測速度場
        """
        # 各項の寄与を計算
        advection = self.advection.compute(velocity, **kwargs)
        diffusion = self.diffusion.compute(velocity, **kwargs)
        force = self.force.compute(velocity, levelset=levelset, **kwargs)

        # 予測速度場を計算
        result = VectorField(velocity.shape, velocity.dx)
        for i in range(velocity.ndim):
            result.components[i].data = velocity.components[i].data + dt * (
                advection[i] + diffusion[i] + force[i]
            )

        return result

    def _pressure_correction_step(
        self, dt: float, velocity: VectorField, pressure: ScalarField, **kwargs
    ) -> ScalarField:
        """圧力補正ステップ

        Args:
            dt: 時間刻み幅
            velocity: 予測速度場
            pressure: 現在の圧力場
            **kwargs: 追加のパラメータ

        Returns:
            圧力補正場
        """
        try:
            # 発散を計算
            div = velocity.divergence()

            # 圧力Poisson方程式の右辺
            # 非圧縮条件: ∇・u^(n+1) = 0 より
            # ∇²p = ρ/dt * ∇・u^*
            density = kwargs.get("density", None)
            if density is not None:
                rhs = density.data * div.data / dt
            else:
                rhs = div.data / dt

            # 圧力補正値の初期推定値
            p_corr = ScalarField(pressure.shape, pressure.dx)

            # Poisson方程式を解く
            p_corr.data = self.poisson_solver.solve(
                initial_solution=np.zeros_like(pressure.data), 
                rhs=rhs, 
                dx=velocity.dx
            )

            # 圧力補正の反復回数を記録
            self._pressure_iterations = self.poisson_solver.iteration_count

            return p_corr

        except Exception as e:
            # エラーログの出力
            if self.logger:
                self.logger.warning(f"圧力補正ステップでエラーが発生: {str(e)}")

            # デフォルトの圧力補正を返す
            p_corr = ScalarField(pressure.shape, pressure.dx)
            p_corr.data = np.zeros_like(pressure.data)
            return p_corr

    def _corrector_step(
        self,
        dt: float,
        velocity: VectorField,
        pressure_correction: ScalarField,
        **kwargs,
    ) -> VectorField:
        """修正子ステップ

        圧力勾配に基づいて速度場を補正します。

        Args:
            dt: 時間刻み幅
            velocity: 予測速度場
            pressure_correction: 圧力補正場
            **kwargs: 追加のパラメータ

        Returns:
            補正された速度場
        """
        return self.pressure.compute_correction(
            velocity, pressure_correction, dt, **kwargs
        )

    def _collect_diagnostics(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        levelset: Optional[LevelSetField],
        divergence: ScalarField,
        **kwargs,
    ) -> Dict[str, Any]:
        """診断情報の収集

        Args:
            velocity: 速度場
            pressure: 圧力場
            levelset: Level Set場
            divergence: 速度発散場
            **kwargs: 追加のパラメータ

        Returns:
            診断情報の辞書
        """
        # 診断情報の収集
        diag = {
            "max_velocity": max(np.max(np.abs(v.data)) for v in velocity.components),
            "max_pressure": np.max(np.abs(pressure.data)),
            "max_divergence": self._max_divergence,
            "pressure_iterations": self._pressure_iterations,
            "kinetic_energy": 0.5
            * sum(np.sum(v.data**2) for v in velocity.components)
            * velocity.dx**velocity.ndim,
        }

        # 各項の診断情報
        diag.update(
            {
                "advection": self.advection.get_diagnostics(velocity, **kwargs),
                "diffusion": self.diffusion.get_diagnostics(velocity, **kwargs),
                "pressure": self.pressure.get_diagnostics(velocity, pressure, **kwargs),
                "force": self.force.get_diagnostics(
                    velocity, levelset=levelset, **kwargs
                ),
            }
        )

        return diag