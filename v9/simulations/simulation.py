"""二相流シミュレーションの主要クラスを提供するモジュール"""

from typing import Dict, Any, Tuple
import numpy as np

from .state import SimulationState
from .initializer import TwoPhaseFlowInitializer
from .config import SimulationConfig
from .checkpoint import CheckpointManager
from .diagnostics import DiagnosticsCollector
from .solvers.navier_stokes_solver import ProjectionNavierStokesSolver

from physics.levelset import LevelSetSolver
from physics.levelset.properties import PropertiesManager


class TwoPhaseFlowSimulator:
    """二相流シミュレーションクラス"""

    def __init__(self, config: SimulationConfig, logger=None):
        """シミュレーションを初期化"""
        self.config = config
        self.logger = logger
        self._state = None
        self._ns_solver = None
        self._ls_solver = None
        self._properties = None

        # 補助コンポーネントの初期化
        self._diagnostics = DiagnosticsCollector()
        self._checkpoint = CheckpointManager(config)

        # 物性値マネージャーの初期化
        self._initialize_properties()

    def _initialize_properties(self):
        """物性値マネージャーを初期化"""
        self._properties = PropertiesManager(
            phase1=self.config.phases["water"].to_properties(),
            phase2=self.config.phases["nitrogen"].to_properties(),
        )

    def initialize(self):
        """シミュレーションを初期化"""
        try:
            # 初期状態の生成
            initializer = TwoPhaseFlowInitializer(
                self.config, self._properties, self.logger
            )
            self._state = initializer.create_initial_state()

            # ソルバーの初期化
            self._initialize_solvers()

            if self.logger:
                self.logger.info(
                    f"シミュレーションを初期化しました\n"
                    f"  格子サイズ: {self.config.domain.dimensions}\n"
                    f"  物理サイズ: {self.config.domain.size} [m]"
                )

        except Exception as e:
            if self.logger:
                self.logger.error(f"初期化中にエラー: {e}")
            raise

    def _initialize_solvers(self):
        """ソルバーを初期化"""
        # Navier-Stokesソルバーの初期化
        self._ns_solver = ProjectionNavierStokesSolver(
            use_weno=self.config.solver.use_weno,
            properties=self._properties,
            logger=self.logger,
        )

        # レベルセットソルバーの初期化
        self._ls_solver = LevelSetSolver(
            use_weno=self.config.solver.use_weno,
            weno_order=self.config.solver.weno_order,
            logger=self.logger,
        )

    def step_forward(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """1時間ステップを進める"""
        if self._state is None:
            raise RuntimeError("シミュレーションが初期化されていません")

        try:
            # 時間刻み幅の計算
            dt = self._compute_timestep()

            # 速度場の時間発展
            ns_result = self._ns_solver.step_forward(state=self._state, dt=dt)
            state_ns = ns_result["velocity"]

            # レベルセット場の時間発展
            ls_result = self._ls_solver.step_forward(state=state_ns, dt=dt)
            state_ls = ls_result["levelset"]

            # 状態の更新
            self._state = SimulationState(
                velocity=state_ns,
                levelset=state_ls,
                pressure=ns_result.get("pressure", self._state.pressure),
                time=ls_result["time"],
                properties=self._properties,
            )

            # 診断情報の収集
            info = self._diagnostics.collect(
                time=self._state.time,
                dt=dt,
                navier_stokes=ns_result.get("diagnostics", {}),
                level_set=ls_result.get("diagnostics", {}),
            )

            return self._state, info

        except Exception as e:
            if self.logger:
                self.logger.error(f"時間発展計算中にエラー: {e}")
            raise

    def _compute_timestep(self) -> float:
        """時間刻み幅を計算"""
        # CFL条件に基づく時間刻み幅の計算
        dt_ns = self._ns_solver.compute_timestep(state=self._state)
        dt_ls = self._ls_solver.compute_timestep(state=self._state)

        # より厳しい制限を採用
        dt = min(dt_ns, dt_ls)

        # 設定された最大・最小値で制限
        return np.clip(dt, self.config.time.min_dt, self.config.time.max_dt)

    def get_state(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """現在のシミュレーション状態を取得"""
        if self._state is None:
            raise RuntimeError("シミュレーションが初期化されていません")

        return self._state, self._diagnostics.get_current()

    def save_checkpoint(self):
        """チェックポイントを保存"""
        if self._state is None:
            raise RuntimeError("シミュレーションが初期化されていません")

        self._checkpoint.save(self._state)

    def load_checkpoint(self):
        """チェックポイントから読み込み"""
        self._state = self._checkpoint.load(self.config, self._properties, self.logger)
