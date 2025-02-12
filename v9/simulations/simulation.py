"""二相流シミュレーションの統合的な実装

このモジュールは、以下の物理過程を統合的に扱います：
1. レベルセット関数からの物性値計算
2. 外力（表面張力・重力）の計算
3. 圧力ポアソン方程式の解法
4. Navier-Stokes方程式の解法
5. レベルセット方程式の時間発展
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from core.field import VectorField, ScalarField
from numerics.time_evolution.euler import ForwardEuler

from physics.levelset import LevelSetField
from physics.navier_stokes import NavierStokesSolver
from physics.pressure import PressurePoissonSolver
from physics.surface_tension import compute_surface_tension_force
from physics.continuity import ContinuityEquation

from simulations.config import SimulationConfig
from simulations.state import SimulationState
from simulations.initializer import SimulationInitializer


class TwoPhaseFlowSimulator:
    """二相流シミュレーションの統合的なソルバー"""

    def __init__(self, config: SimulationConfig):
        """シミュレータを初期化

        Args:
            config: シミュレーション設定
        """
        self.config = config

        # 表面張力係数の計算
        surface_tension_coefficient = 0.0
        if len(config.physics.phases) >= 2:
            surface_tension_coefficient = abs(
                config.physics.phases[0].surface_tension
                - config.physics.phases[1].surface_tension
            )

        # デフォルトの時間積分器
        self._time_solver = ForwardEuler(
            cfl=config.numerical.cfl,
            min_dt=config.numerical.min_dt,
            max_dt=config.numerical.max_dt,
        )

        # ソルバーの初期化
        self._navier_stokes_solver = NavierStokesSolver()
        self._pressure_solver = PressurePoissonSolver()
        self._continuity_solver = ContinuityEquation()

        # 表面張力係数の保存
        self._surface_tension_coefficient = surface_tension_coefficient

        # 現在の状態
        self._current_state = None
        self._diagnostics = {}

    def compute_material_properties(
        self, levelset: LevelSetField
    ) -> Dict[str, ScalarField]:
        """レベルセット関数から物性値を計算

        Args:
            levelset: レベルセット関数

        Returns:
            密度場、粘性場、界面関数を含む辞書
        """
        density = ScalarField(levelset.shape, levelset.dx)
        viscosity = ScalarField(levelset.shape, levelset.dx)

        # Heaviside関数を使用して物性値を計算
        heaviside = levelset.get_heaviside()
        density.data = heaviside.data
        viscosity.data = heaviside.data

        # デルタ関数を使用して界面を特定
        delta = levelset.get_delta()
        interface = ScalarField(levelset.shape, levelset.dx)
        interface.data = delta.data

        return {"density": density, "viscosity": viscosity, "interface": interface}

    def compute_forces(
        self,
        velocity: VectorField,
        density: ScalarField,
        levelset: LevelSetField,
    ) -> VectorField:
        """外力（重力と表面張力）を計算"""
        # 重力項
        gravity_force = VectorField(density.shape, density.dx)
        gravity_force.components[-1].data = -self.config.physics.gravity * density.data

        # 表面張力項の計算
        surface_tension_force, surface_tension_info = compute_surface_tension_force(
            levelset, surface_tension_coefficient=self._surface_tension_coefficient
        )

        # 外力を統合
        total_force = VectorField(density.shape, density.dx)
        for i in range(velocity.ndim):
            total_force.components[i].data = (
                gravity_force.components[i].data
                + surface_tension_force.components[i].data
            )

        # 診断情報の更新
        self._diagnostics.update(
            {
                "surface_tension": surface_tension_info,
                "gravity_force_max": float(
                    np.max(np.abs(gravity_force.components[-1].data))
                ),
            }
        )

        return total_force

    def compute_derivative(self, state: SimulationState) -> SimulationState:
        """状態の時間微分を計算

        Args:
            state: 現在の状態

        Returns:
            時間微分を表す新しい状態
        """
        # Level Set場の時間発展を計算（移流方程式）
        levelset_derivative = self._continuity_solver.compute_derivative(
            state.levelset, state.velocity
        )

        # 密度と粘性を計算
        density = state.get_density()
        viscosity = state.get_viscosity()

        # 加速度項を計算
        acceleration = self._navier_stokes_solver.compute_velocity_derivative(
            velocity=state.velocity,
            density=density,
            viscosity=viscosity,
            pressure=state.pressure,
            levelset=state.levelset,
        )

        # 新しい状態を初期化
        derivative_state = SimulationState(
            time=0.0,
            velocity=VectorField(state.velocity.shape, state.velocity.dx),
            levelset=LevelSetField(shape=state.levelset.shape, dx=state.levelset.dx),
            pressure=ScalarField(state.pressure.shape, state.pressure.dx),
        )

        # 時間微分を設定
        derivative_state.levelset.data = levelset_derivative
        derivative_state.velocity.components = [
            ScalarField(v.shape, v.dx, initial_value=a)
            for v, a in zip(state.velocity.components, acceleration)
        ]

        return derivative_state

    def step_forward(
        self, state: Optional[SimulationState] = None, dt: Optional[float] = None
    ) -> Tuple[SimulationState, Dict[str, Any]]:
        """シミュレーションを1ステップ進める"""
        if state is None:
            state = self._current_state
            if state is None:
                raise ValueError("シミュレーション状態が初期化されていません")

        # 時間刻み幅の計算
        if dt is None:
            dt = self._time_solver.compute_timestep(state=state)

        # 1. 状態の時間発展を計算
        new_state = self._time_solver.integrate(
            state, dt, lambda s: self.compute_derivative(s)
        )

        # Level Set関数の再初期化（必要に応じて）
        if self._should_reinitialize(new_state):
            new_state.levelset.reinitialize()

        # 診断情報の更新
        self._diagnostics.update(
            {
                "time": new_state.time,
                "dt": dt,
            }
        )

        # 現在の状態を更新
        self._current_state = new_state

        return new_state, self._diagnostics

    def _should_reinitialize(self, state: SimulationState) -> bool:
        """Level Set関数の再初期化が必要か判定"""
        reinit_interval = self.config.numerical.level_set_reinit_interval
        if reinit_interval <= 0:
            return False

        current_step = int(state.time / self._time_solver.dt)
        return current_step % reinit_interval == 0

    def initialize(self, state: Optional[SimulationState] = None):
        """シミュレーションを初期化"""
        if state is None:
            initializer = SimulationInitializer(self.config)
            state = initializer.create_initial_state()
        self._current_state = state

    def get_state(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """現在のシミュレーション状態を取得"""
        if self._current_state is None:
            raise ValueError("シミュレーション状態が初期化されていません")
        return self._current_state, self._current_state.get_diagnostics()

    def save_checkpoint(self, filepath: str):
        """チェックポイントを保存"""
        if self._current_state is None:
            raise ValueError("シミュレーション状態が初期化されていません")
        self._current_state.save_state(filepath)

    def load_checkpoint(self, filepath: str) -> SimulationState:
        """チェックポイントから状態を読み込み"""
        return SimulationState.load_state(filepath)