"""
二相流シミュレーションの統合的な実装

このモジュールは、Level Set法を用いた二相流体シミュレーションの
物理過程を統合的に取り扱います。

主な機能:
1. レベルセット関数からの物性値計算
2. 外力（表面張力・重力）の計算
3. 圧力ポアソン方程式の解法
4. Navier-Stokes方程式の解法
5. レベルセット方程式の時間発展
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import numpy as np

from core.field import VectorField, ScalarField
from core.solver import TemporalSolver
from numerics.time_evolution import ForwardEuler, RungeKutta4

from physics.levelset import LevelSetField
from physics.navier_stokes import NavierStokesSolver
from physics.pressure import PressurePoissonSolver
from physics.surface_tension import SurfaceTensionCalculator
from physics.continuity import ContinuityEquation

from simulations.config import SimulationConfig
from simulations.state import SimulationState
from simulations.initializer import SimulationInitializer


class TwoPhaseFlowSimulator:
    """
    二相流シミュレーションの統合的なソルバー

    複雑な物理過程を統合的に解くための高度なシミュレーションフレームワーク。
    """

    def __init__(
        self, config: SimulationConfig, time_integrator: Optional[TemporalSolver] = None
    ):
        """
        シミュレータを初期化

        Args:
            config: シミュレーション設定
            time_integrator: 時間積分器（指定しない場合は自動選択）
        """
        self.config = config
        self._state: Optional[SimulationState] = None
        self._diagnostics: Dict[str, Any] = {}

        # 時間積分器の選択
        if time_integrator is None:
            time_integrator_name = config.numerical.time_integrator.lower()
            if time_integrator_name == "rk4":
                self._time_solver = RungeKutta4(
                    cfl=config.numerical.cfl,
                    min_dt=config.numerical.min_dt,
                    max_dt=config.numerical.max_dt,
                )
            else:
                self._time_solver = ForwardEuler(
                    cfl=config.numerical.cfl,
                    min_dt=config.numerical.min_dt,
                    max_dt=config.numerical.max_dt,
                )
        else:
            self._time_solver = time_integrator

        # ソルバーの初期化
        self._setup_solvers()

    def _setup_solvers(self):
        """
        物理計算用のソルバーを初期化
        """
        # 表面張力係数の計算
        surface_tension_coeff = 0.0
        if len(self.config.physics.phases) >= 2:
            surface_tension_coeff = abs(
                self.config.physics.phases[0].surface_tension
                - self.config.physics.phases[1].surface_tension
            )

        # ソルバーの初期化
        self._navier_stokes_solver = NavierStokesSolver()
        self._pressure_solver = PressurePoissonSolver()
        self._continuity_solver = ContinuityEquation()
        self._surface_tension_calculator = SurfaceTensionCalculator(
            surface_tension_coefficient=surface_tension_coeff
        )

    def compute_material_properties(
        self, levelset: LevelSetField
    ) -> Tuple[ScalarField, ScalarField]:
        """
        レベルセット関数から物性値を計算

        Args:
            levelset: レベルセット関数場

        Returns:
            (密度場, 粘性場)のタプル
        """
        # Heaviside関数を使用して物性値を計算
        density = levelset.get_density(
            rho1=self.config.physics.phases[0].density,
            rho2=self.config.physics.phases[1].density,
        )

        viscosity = levelset.get_density(
            rho1=self.config.physics.phases[0].viscosity,
            rho2=self.config.physics.phases[1].viscosity,
        )

        return density, viscosity

    def compute_external_forces(
        self,
        levelset: LevelSetField,
        density: ScalarField,
    ) -> Tuple[VectorField, Dict[str, Any]]:
        """
        外力（重力と表面張力）を計算

        Args:
            levelset: レベルセット関数場
            density: 密度場

        Returns:
            (外力のベクトル場, 診断情報)のタプル
        """
        # 重力項の計算
        gravity_force = VectorField(levelset.shape, levelset.dx)
        for i, comp in enumerate(gravity_force.components):
            # 最後の次元（z方向）にのみ重力を適用
            if i == len(levelset.shape) - 1:
                comp.data = -self.config.physics.gravity * density.data
            else:
                comp.data = np.zeros(density.shape)

        # 表面張力項の計算
        surface_tension_force, surface_tension_info = (
            self._surface_tension_calculator.compute_force(levelset=levelset)
        )

        # 外力を統合
        total_force = gravity_force + surface_tension_force

        # 診断情報の更新
        diagnostics = {
            "surface_tension": surface_tension_info,
            "gravity_max": float(np.max(np.abs(gravity_force.components[-1].data))),
            "total_force_max": float(
                max(np.max(np.abs(f.data)) for f in total_force.components)
            ),
        }

        return total_force, diagnostics

    def compute_pressure(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        external_force: VectorField,
    ) -> Tuple[ScalarField, Dict[str, Any]]:
        """
        圧力場を計算

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            external_force: 外力場

        Returns:
            (圧力場, 診断情報)のタプル
        """
        # 圧力場の計算
        pressure, solver_diagnostics = self._pressure_solver.solve(
            density=density.data,
            velocity=velocity,
            viscosity=viscosity,
            external_force=external_force,
        )

        # 診断情報の更新
        self._diagnostics.update(solver_diagnostics)

        return pressure, solver_diagnostics

    def step_forward(
        self, state: Optional[SimulationState] = None, dt: Optional[float] = None
    ) -> Tuple[SimulationState, Dict[str, Any]]:
        """
        シミュレーションを1ステップ進める

        Args:
            state: 現在の状態（Noneの場合は内部状態を使用）
            dt: 時間刻み幅（Noneの場合は自動計算）

        Returns:
            (更新された状態, 診断情報)のタプル
        """
        # 状態の確認と取得
        if state is None:
            state = self._state
            if state is None:
                raise ValueError("シミュレーション状態が初期化されていません")

        # 時間刻み幅の計算
        if dt is None:
            dt = self._time_solver.compute_timestep(state=state)

        # 物性値の計算
        density, viscosity = self.compute_material_properties(levelset=state.levelset)

        # 外力の計算
        external_forces, force_diagnostics = self.compute_external_forces(
            levelset=state.levelset,
            density=density,
        )

        # 圧力場の計算
        pressure, pressure_diagnostics = self.compute_pressure(
            velocity=state.velocity,
            density=density,
            viscosity=viscosity,
            external_force=external_forces,
        )

        # 速度とレベルセットの時間微分を計算
        velocity_derivative = self._navier_stokes_solver.compute(
            velocity=state.velocity,
            density=density,
            viscosity=viscosity,
            pressure=pressure,
            external_force=external_forces,
        )

        levelset_derivative = self._continuity_solver.compute_derivative(
            field=state.levelset, velocity=state.velocity
        )

        # 時間積分器による更新
        new_velocity = self._time_solver.integrate(
            state.velocity, dt, velocity_derivative
        )
        new_levelset = self._time_solver.integrate(
            state.levelset, dt, levelset_derivative
        )

        # レベルセット関数の再初期化
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
        self._state = new_state

        return new_state, diagnostics

    def _should_reinitialize(self, state: SimulationState) -> bool:
        """
        Level Set関数の再初期化が必要か判定

        Args:
            state: シミュレーション状態

        Returns:
            再初期化が必要かどうか
        """
        reinit_interval = self.config.numerical.level_set.get("reinit_interval", 10)
        if reinit_interval <= 0:
            return False

        # 時間ベースでの再初期化チェック
        current_step = int(state.time / self.config.numerical.initial_dt)
        return current_step % reinit_interval == 0

    def initialize(self, state: Optional[SimulationState] = None):
        """
        シミュレーションを初期化

        Args:
            state: 初期状態（Noneの場合は設定から自動生成）
        """
        if state is None:
            initializer = SimulationInitializer(self.config)
            state = initializer.create_initial_state()

        self._state = state

    def get_state(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """
        現在のシミュレーション状態を取得

        Returns:
            (現在の状態, 診断情報)のタプル
        """
        if self._state is None:
            raise ValueError("シミュレーション状態が初期化されていません")

        return self._state, self._state.get_diagnostics()

    def save_checkpoint(self, filepath: str):
        """
        現在の状態をチェックポイントとして保存

        Args:
            filepath: 保存先のファイルパス
        """
        if self._state is None:
            raise ValueError("シミュレーション状態が初期化されていません")

        self._state.save_state(filepath)

    def load_checkpoint(self, filepath: str) -> SimulationState:
        """
        チェックポイントから状態を読み込み

        Args:
            filepath: 読み込むチェックポイントファイルのパス

        Returns:
            読み込まれた状態
        """
        return SimulationState.load_state(filepath)


# 将来の拡張性のためのファクトリメソッド
def create_simulator(
    config: SimulationConfig, time_integrator: Optional[TemporalSolver] = None
) -> TwoPhaseFlowSimulator:
    """
    シミュレータのファクトリメソッド

    Args:
        config: シミュレーション設定
        time_integrator: オプションの時間積分器

    Returns:
        初期化されたTwoPhaseFlowSimulatorインスタンス
    """
    simulator = TwoPhaseFlowSimulator(config, time_integrator)
    return simulator
