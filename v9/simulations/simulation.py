"""二相流シミュレーションの主要クラスを提供するモジュール

リファクタリングされたphysics/パッケージに対応した更新版
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

from physics.levelset import LevelSetPropertiesManager, LevelSetSolver
from physics.navier_stokes import solvers, terms
from physics.time_evolution import TimeEvolutionSolver


from .config import SimulationConfig
from .state import SimulationState
from .initializer import SimulationInitializer


class TwoPhaseFlowSimulator:
    """二相流シミュレーションクラス

    リファクタリングされたphysics/パッケージを活用した
    高度な二相流シミュレーションを実現
    """

    def __init__(self, config: SimulationConfig, logger=None):
        """シミュレーションを初期化

        Args:
            config: シミュレーション設定
            logger: ロガー（オプション）
        """
        self.config = config
        self.logger = logger

        # 物性値マネージャーの初期化
        self._properties = LevelSetPropertiesManager(
            phase1=config.phases["water"].to_properties(),
            phase2=config.phases["nitrogen"].to_properties(),
        )

        # ソルバーコンポーネントの初期化
        self._initialize_solvers()

    def _initialize_solvers(self):
        """ソルバーコンポーネントを初期化"""
        # 移流項
        advection_term = terms.AdvectionTerm(
            use_weno=self.config.solver.use_weno,
            weno_order=self.config.solver.weno_order,
        )

        # 粘性項
        diffusion_term = terms.DiffusionTerm(use_conservative=True)

        # 外力項（重力など）
        gravity_force = terms.GravityForce(gravity=self.config.physics.gravity)
        force_term = terms.ForceTerm(forces=[gravity_force])

        # 圧力項
        pressure_term = terms.PressureTerm()

        # Navier-Stokesソルバーの設定
        self._ns_solver = solvers.ProjectionSolver(
            terms=[advection_term, diffusion_term, force_term, pressure_term],
            properties=self._properties,
            use_rotational=True,
        )

        # Level Setソルバーの設定
        self._ls_solver = LevelSetSolver(
            use_weno=self.config.solver.use_weno,
            weno_order=self.config.solver.weno_order,
        )

        # 時間発展ソルバー
        self._time_solver = TimeEvolutionSolver(
            terms=[self._ns_solver, self._ls_solver],
            integrator_type=self.config.solver.time_integrator,
        )

    def initialize(self, state: Optional[SimulationState] = None):
        """シミュレーションを初期化

        Args:
            state: 初期状態（オプション）
        """
        if state is None:
            initializer = SimulationInitializer(self.config)
            state = initializer.create_initial_state()

        self._time_solver.initialize(state)
        self._current_state = state  # 現在の状態を保持

    def get_state(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """現在のシミュレーション状態を取得

        Returns:
            現在の状態と診断情報のタプル
        """
        return self._current_state, {
            "time": self._time_solver.time,
            "dt": self._time_solver.dt,
            "iteration": self._time_solver.iteration_count,
        }

    def step_forward(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """1時間ステップを進める

        Returns:
            更新された状態と診断情報
        """
        return self._time_solver.step_forward()

    def save_checkpoint(self, filepath: str):
        """シミュレーション状態をチェックポイントとして保存

        Args:
            filepath: 保存先のファイルパス
        """
        # チェックポイントディレクトリを作成
        output_dir = Path(filepath).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # 現在の状態を取得
        state, diagnostics = self.get_state()

        # NumPyを使用してチェックポイントを保存
        np.savez(
            filepath,
            velocity_components=[v.data for v in state.velocity.components],
            velocity_shape=state.velocity.shape,
            velocity_dx=state.velocity.dx,
            levelset_data=state.levelset.data,
            levelset_shape=state.levelset.shape,
            levelset_dx=state.levelset.dx,
            pressure_data=state.pressure.data,
            pressure_shape=state.pressure.shape,
            pressure_dx=state.pressure.dx,
            simulation_time=state.time,
            # 追加のメタデータ
            diagnostics=diagnostics,
        )

        # ロギング
        if self.logger:
            self.logger.info(f"チェックポイントを保存: {filepath}")

    def load_checkpoint(self, filepath: str):
        """チェックポイントから状態を復元

        Args:
            filepath: 読み込むチェックポイントファイルのパス

        Returns:
            復元された状態
        """
        try:
            # チェックポイントファイルを読み込み
            with np.load(filepath, allow_pickle=True) as checkpoint:
                # 速度場の復元
                from core.field import VectorField

                velocity = VectorField(
                    tuple(checkpoint["velocity_shape"]), checkpoint["velocity_dx"]
                )
                for i, comp in enumerate(velocity.components):
                    comp.data = checkpoint["velocity_components"][i]

                # レベルセット場の復元
                from physics.levelset import LevelSetField

                levelset = LevelSetField(
                    tuple(checkpoint["levelset_shape"]), checkpoint["levelset_dx"]
                )
                levelset.data = checkpoint["levelset_data"]

                # 圧力場の復元
                from core.field import ScalarField

                pressure = ScalarField(
                    tuple(checkpoint["pressure_shape"]), checkpoint["pressure_dx"]
                )
                pressure.data = checkpoint["pressure_data"]

                # シミュレーション状態の復元
                from .state import SimulationState

                state = SimulationState(
                    velocity=velocity,
                    levelset=levelset,
                    pressure=pressure,
                    time=float(checkpoint["simulation_time"]),
                    properties=self._properties,
                )

                # ロギング
                if self.logger:
                    self.logger.info(f"チェックポイントから復元: {filepath}")

                return state

        except Exception as e:
            if self.logger:
                self.logger.error(f"チェックポイントの読み込み中にエラー: {e}")
            raise
