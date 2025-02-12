"""
二相流シミュレーションの統合的な実装

このモジュールは、二相流シミュレーションの物理過程を統合的に扱います：
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
            levelset: レベルセット関数場

        Returns:
            密度場、粘性場を含む辞書
        """
        # Heaviside関数を使用して物性値を計算
        density = levelset.get_density(
            rho1=self.config.physics.phases[0].density,
            rho2=self.config.physics.phases[1].density,
        )

        # 粘性場も同様に計算
        viscosity = ScalarField(levelset.shape, levelset.dx)
        viscosity.data = levelset.get_density(
            rho1=self.config.physics.phases[0].viscosity,
            rho2=self.config.physics.phases[1].viscosity,
        ).data

        return {"density": density, "viscosity": viscosity}

    def compute_external_forces(
        self,
        levelset: LevelSetField,
    ) -> Tuple[VectorField, Dict[str, Any]]:
        """外力（重力と表面張力）を計算

        Args:
            levelset: レベルセット関数場

        Returns:
            外力のベクトル場と関連する診断情報
        """
        # 重力項
        gravity_force = VectorField(levelset.shape, levelset.dx)
        gravity_direction = [
            -self.config.physics.gravity if axis == len(levelset.shape) - 1 else 0
            for axis in range(levelset.ndim)
        ]
        for i, comp in enumerate(gravity_force.components):
            comp.data = gravity_direction[i]

        # 表面張力項の計算
        surface_tension_force, surface_tension_info = compute_surface_tension_force(
            levelset, surface_tension_coefficient=self._surface_tension_coefficient
        )

        # 外力を統合
        total_force = VectorField(levelset.shape, levelset.dx)
        for i in range(levelset.ndim):
            total_force.components[i].data = (
                gravity_force.components[i].data
                + surface_tension_force.components[i].data
            )

        # 診断情報の更新
        diagnostics = {
            "surface_tension": surface_tension_info,
            "gravity_max": float(np.max(np.abs(gravity_force.components[-1].data))),
            "total_force_max": float(
                np.max([np.abs(f.data).max() for f in total_force.components])
            ),
        }
        self._diagnostics.update(diagnostics)

        return total_force, diagnostics

    def compute_pressure(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        external_force: VectorField,
    ) -> Tuple[ScalarField, Dict[str, Any]]:
        """圧力場を計算

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            external_force: 外力場

        Returns:
            計算された圧力場と診断情報
        """
        # 圧力場の計算
        pressure = ScalarField(velocity.shape, velocity.dx)
        pressure.data, solver_diagnostics = self._pressure_solver.solve(
            velocity=velocity, density=density, viscosity=viscosity
        )

        # 診断情報の更新
        self._diagnostics.update(solver_diagnostics)

        return pressure, solver_diagnostics

    def step_forward(
        self, state: Optional[SimulationState] = None, dt: Optional[float] = None
    ) -> Tuple[SimulationState, Dict[str, Any]]:
        """シミュレーションを1ステップ進める

        Args:
            state: 現在の状態（Noneの場合は内部状態を使用）
            dt: 時間刻み幅（Noneの場合は自動計算）

        Returns:
            (更新された状態, 診断情報)のタプル
        """
        if state is None:
            state = self._current_state
            if state is None:
                raise ValueError("シミュレーション状態が初期化されていません")

        # 時間刻み幅の計算
        if dt is None:
            dt = self._time_solver.compute_timestep(state=state)

        # 物性値の計算
        material_properties = self.compute_material_properties(state.levelset)

        # 外力の計算
        external_forces, force_diagnostics = self.compute_external_forces(
            state.velocity, state.levelset
        )

        # 圧力場の計算
        pressure, pressure_diagnostics = self.compute_pressure(
            state.velocity,
            material_properties["density"],
            material_properties["viscosity"],
            external_forces,
        )

        # 速度場の時間微分を計算
        velocity_derivative = self._navier_stokes_solver.compute_velocity_derivative(
            state.velocity,
            material_properties["density"],
            material_properties["viscosity"],
            pressure,
            state.levelset,
        )

        # レベルセット関数の時間微分を計算
        levelset_derivative = self._continuity_solver.compute_derivative(
            state.levelset, state.velocity
        )

        # 時間積分器による更新
        new_velocity = self._time_solver.integrate(
            state.velocity, dt, lambda v: VectorField.from_list(velocity_derivative)
        )
        new_levelset = self._time_solver.integrate(
            state.levelset, dt, lambda l: levelset_derivative
        )

        # 必要に応じてレベルセット関数の再初期化
        if self._should_reinitialize(new_levelset):
            new_levelset.reinitialize()

        # 新しい状態の作成
        new_state = SimulationState(
            time=state.time + dt,
            velocity=new_velocity,
            levelset=new_levelset,
            pressure=pressure,
        )

        # 診断情報の更新
        diagnostics = {
            **force_diagnostics,
            **pressure_diagnostics,
            "time": new_state.time,
            "dt": dt,
        }
        self._diagnostics = diagnostics

        # 現在の状態を更新
        self._current_state = new_state

        return new_state, diagnostics

    def _should_reinitialize(self, state: SimulationState) -> bool:
        """Level Set関数の再初期化が必要か判定

        Args:
            state: シミュレーション状態

        Returns:
            再初期化が必要かどうか
        """
        reinit_interval = self.config.numerical.get("level_set_reinit_interval", 10)
        if reinit_interval <= 0:
            return False

        # 時間ベースでの再初期化チェック
        current_step = int(state.time / self.config.numerical.initial_dt)
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
