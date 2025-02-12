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

        # 1. 物性値の計算
        material_properties = self.compute_material_properties(state.levelset)
        density = material_properties["density"]
        viscosity = material_properties["viscosity"]

        # 2. 外力の計算
        external_forces = self.compute_forces(state.velocity, density, state.levelset)

        # 3. 圧力場の計算
        pressure, pressure_info = self._pressure_solver.solve(
            velocity=state.velocity,
            density=density,
            viscosity=viscosity,
            dt=dt,
            external_force=external_forces,
        )

        # 4. 速度と界面の時間発展
        new_state = self._time_solver.integrate(
            state, dt, lambda s: s.compute_derivative()
        )

        # Level Set関数の再初期化（必要に応じて）
        if self._should_reinitialize(new_state):
            new_state.levelset.reinitialize()

        # 診断情報の更新
        self._diagnostics.update(
            {
                "time": new_state.time,
                "dt": dt,
                "pressure": pressure_info,
                "material_properties": {
                    "density_range": (
                        float(np.min(density.data)),
                        float(np.max(density.data)),
                    ),
                    "viscosity_range": (
                        float(np.min(viscosity.data)),
                        float(np.max(viscosity.data)),
                    ),
                },
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
