"""シミュレーションの全体的な管理を担当するモジュール"""

from physics.levelset import LevelSetMethod
from physics.navier_stokes.solvers.projection import PressureProjectionSolver
from numerics.time_evolution.euler import ForwardEuler
from .config import SimulationConfig
from .state import SimulationState
from .initializer import SimulationInitializer
from physics.navier_stokes.terms import (
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
    AccelerationTerm,
    GravityForce,
    SurfaceTensionForce,
)
from numerics.poisson import PoissonConfig
from typing import Optional, Dict, Any


class TwoPhaseFlowSimulator:
    """二相流シミュレーションを管理するクラス"""

    def __init__(self, config: SimulationConfig):
        """
        シミュレーションを初期化

        Args:
            config: シミュレーション設定
        """
        self.config = config

        # ポアソンソルバーの設定
        poisson_config = PoissonConfig()
        poisson_config.convergence["tolerance"] = 1e-5
        poisson_config.convergence["max_iterations"] = 1000

        # 時間積分器の設定
        self._time_solver = ForwardEuler(
            cfl=config.numerical.cfl or 0.5,
        )

        # Navier-Stokes方程式の各項を設定
        self.terms = [
            AdvectionTerm(use_weno=True),
            DiffusionTerm(viscosity=self._get_viscosity()),
            PressureTerm(density=self._get_density()),
            GravityForce(),
            SurfaceTensionForce(),
        ]

        # 加速度項と時間積分器
        self.acceleration_term = AccelerationTerm()

        # Level Set法と圧力投影法のソルバー
        self.levelset_method = LevelSetMethod()
        self.navier_stokes_solver = PressureProjectionSolver(poisson_config)

        # シミュレーション状態の初期化用フラグ
        self._state = None
        self._initialized = False

    def _get_viscosity(self) -> float:
        """流体の粘性係数を取得"""
        phases = self.config.phases
        return min(phase.viscosity for phase in phases.values())

    def _get_density(self) -> float:
        """流体の参照密度を取得"""
        phases = self.config.phases
        return min(phase.density for phase in phases.values())

    def initialize(self, state: Optional[SimulationState] = None):
        """
        シミュレーションを初期化

        Args:
            state: 初期状態（オプション）
        """
        if state is None:
            # 初期状態の生成
            initializer = SimulationInitializer(self.config)
            state = initializer.create_initial_state()

        self._state = state
        self._initialized = True

    def load_checkpoint(self, filepath: str) -> SimulationState:
        """
        チェックポイントファイルから状態を読み込み

        Args:
            filepath: チェックポイントファイルのパス

        Returns:
            読み込まれた状態
        """
        return SimulationState.load_state(filepath)

    def save_checkpoint(self, filepath: str):
        """
        現在の状態をチェックポイントファイルに保存

        Args:
            filepath: 保存先のファイルパス
        """
        if self._state is None:
            raise RuntimeError("シミュレーション状態が初期化されていません")

        self._state.save_state(filepath)

    def get_state(self):
        """
        現在のシミュレーション状態を取得

        Returns:
            現在の状態と追加情報
        """
        if not self._initialized:
            raise RuntimeError("シミュレーションが初期化されていません")

        return self._state, {"time": self._state.time}

    def compute_timestep(self, state: Optional[SimulationState] = None, **kwargs) -> float:
        """
        時間刻み幅を計算

        Args:
            state: 現在の状態（Noneの場合は現在のシミュレーション状態を使用）
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅
        """
        if not self._initialized:
            raise RuntimeError("シミュレーションが初期化されていません")

        # 状態の設定
        current_state = state or self._state

        # 各項から時間刻み幅の上限を取得
        dt_terms = [
            term.compute_timestep(current_state.velocity) for term in self.terms
        ]
        
        # CFL条件に基づいた時間刻み幅
        return min(dt_terms)

    def step_forward(self, dt: Optional[float] = None, state: Optional[SimulationState] = None, **kwargs) -> Dict[str, Any]:
        """
        シミュレーションを1ステップ進める

        Args:
            dt: 時間刻み幅（Noneの場合は自動計算）
            state: 初期状態（Noneの場合は現在の状態を使用）
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態と診断情報
        """
        if not self._initialized:
            raise RuntimeError("シミュレーションが初期化されていません")

        # 状態の設定
        current_state = state or self._state

        # 時間刻み幅の計算
        if dt is None:
            dt = self.compute_timestep(current_state)

        # Level Set関数の移流
        levelset_derivative = self.levelset_method.run(
            current_state.levelset, current_state.velocity
        )
        new_levelset_data = self._time_solver.integrate(
            current_state.levelset.data, dt, derivative_fn=lambda x: levelset_derivative
        )
        new_levelset = current_state.levelset.__class__(
            data=new_levelset_data, dx=current_state.levelset.dx
        )

        # 密度と粘性の計算
        density = new_levelset.heaviside()
        viscosity = new_levelset.heaviside()

        # 加速度項の計算
        acceleration = self.acceleration_term.compute(
            velocity=current_state.velocity,
            density=density,
            viscosity=viscosity,
            pressure=current_state.pressure,
        )

        # 速度場の更新
        new_velocity_comps = []
        for i, (v, a) in enumerate(
            zip(current_state.velocity.components, acceleration)
        ):
            new_v_data = self._time_solver.integrate(v.data, dt, derivative_fn=lambda x: a)
            new_velocity_comps.append(new_v_data)

        new_velocity = current_state.velocity.__class__(
            shape=current_state.velocity.shape, dx=current_state.velocity.dx
        )
        for i, comp in enumerate(new_velocity_comps):
            new_velocity.components[i].data = comp

        # 圧力場の計算
        rhs = self.navier_stokes_solver.compute_rhs(
            velocity=new_velocity, density=density, viscosity=viscosity, dt=dt
        )
        pressure = self.navier_stokes_solver.solve_pressure(rhs, new_velocity)
        new_velocity = self.navier_stokes_solver.project_velocity(
            new_velocity, pressure
        )

        # 新しい状態の作成
        new_state = SimulationState(
            time=current_state.time + dt,
            velocity=new_velocity,
            levelset=new_levelset,
            pressure=pressure,
        )

        # 診断情報の更新
        diagnostics = {
            "time": new_state.time,
            "dt": dt,
            "navier_stokes": self.navier_stokes_solver.get_diagnostics(),
            "levelset": {
                "volume": new_levelset.volume(),
                "area": new_levelset.area(),
            },
        }

        # 状態の更新
        self._state = new_state

        return {"state": new_state, "diagnostics": diagnostics}