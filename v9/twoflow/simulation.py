"""二相流シミュレーションの主要クラスを提供するモジュール

このモジュールは、二相流シミュレーション全体を管理する中心的なクラスを提供します。
速度場とレベルセット場の時間発展を統合的に管理し、初期条件の設定から
シミュレーションの実行まで一貫して制御します。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np

from physics.levelset import LevelSetField
from physics.properties import PropertiesManager
from core.field import VectorField, ScalarField
from physics.navier_stokes import NavierStokesSolver
from physics.levelset import LevelSetSolver
from .config import SimulationConfig


@dataclass
class SimulationState:
    """シミュレーションの状態を保持するクラス"""
    velocity: VectorField
    levelset: LevelSetField
    pressure: ScalarField
    time: float = 0.0
    properties: Optional[PropertiesManager] = None

    def copy(self) -> 'SimulationState':
        """状態の深いコピーを作成"""
        return SimulationState(
            velocity=self.velocity.copy(),
            levelset=self.levelset.copy(),
            pressure=self.pressure.copy(),
            time=self.time,
            properties=self.properties  # PropertiesManagerは共有して問題ない
        )


class TwoPhaseFlowSimulation:
    """二相流シミュレーションクラス"""

    def __init__(
        self,
        config: SimulationConfig,
        logger=None
    ):
        """シミュレーションを初期化

        Args:
            config: シミュレーション設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger
        self._state = None
        self._ns_solver = None
        self._ls_solver = None
        self._properties = None

        # 物性値マネージャーの初期化
        self._initialize_properties()

    def _initialize_properties(self):
        """物性値マネージャーを初期化"""
        self._properties = PropertiesManager(
            phase1=self.config.phases["water"].to_properties(),
            phase2=self.config.phases["nitrogen"].to_properties()
        )

    def initialize(self):
        """シミュレーションを初期化"""
        try:
            # 初期状態の生成
            self._state = self._create_initial_state()

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
        self._ns_solver = NavierStokesSolver(
            logger=self.logger,
            use_weno=self.config.solver.use_weno
        )
        self._ns_solver.initialize()

        # レベルセットソルバーの初期化
        self._ls_solver = LevelSetSolver(
            use_weno=self.config.solver.use_weno,
            weno_order=self.config.solver.weno_order,
            logger=self.logger
        )
        self._ls_solver.initialize()

    def step_forward(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """1時間ステップを進める

        Returns:
            (更新された状態, 計算情報を含む辞書)のタプル
        """
        if self._state is None:
            raise RuntimeError("シミュレーションが初期化されていません")

        try:
            # 時間刻み幅の計算
            dt = self._compute_timestep()

            # 速度場の時間発展
            state_ns, ns_info = self._ns_solver.step_forward(
                self._state,
                dt
            )

            # レベルセット場の時間発展
            state_ls, ls_info = self._ls_solver.step_forward(
                state_ns,
                dt
            )

            # 状態の更新
            self._state = state_ls
            self._state.time += dt

            # 診断情報の収集
            info = {
                "time": self._state.time,
                "dt": dt,
                "navier_stokes": ns_info,
                "level_set": ls_info
            }

            return self._state, info

        except Exception as e:
            if self.logger:
                self.logger.error(f"時間発展計算中にエラー: {e}")
            raise

    def _compute_timestep(self) -> float:
        """時間刻み幅を計算"""
        # CFL条件に基づく時間刻み幅の計算
        dt_ns = self._ns_solver.compute_timestep(self._state)
        dt_ls = self._ls_solver.compute_timestep(self._state)

        # より厳しい制限を採用
        dt = min(dt_ns, dt_ls)

        # 設定された最大・最小値で制限
        return np.clip(
            dt,
            self.config.time.min_dt,
            self.config.time.max_dt
        )

    def save_checkpoint(self, filepath: Path):
        """チェックポイントを保存

        Args:
            filepath: 保存先のパス
        """
        if self._state is None:
            raise RuntimeError("シミュレーションが初期化されていません")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._state.save_to_file(str(filepath))

    @classmethod
    def load_checkpoint(
        cls,
        config: SimulationConfig,
        checkpoint_path: Path,
        logger=None
    ) -> 'TwoPhaseFlowSimulation':
        """チェックポイントから読み込み

        Args:
            config: シミュレーション設定
            checkpoint_path: チェックポイントファイルのパス
            logger: ロガー

        Returns:
            読み込まれたシミュレーション
        """
        sim = cls(config, logger)
        sim._state = SimulationState.load_from_file(str(checkpoint_path))
        sim._initialize_properties()
        sim._initialize_solvers()
        return sim

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        if self._state is None:
            return {"status": "not_initialized"}

        return {
            "time": self._state.time,
            "velocity": {
                "max": float(max(
                    np.max(np.abs(c.data))
                    for c in self._state.velocity.components
                )),
                "kinetic_energy": float(sum(
                    np.sum(c.data**2)
                    for c in self._state.velocity.components
                ) * 0.5 * self._state.velocity.dx**3)
            },
            "pressure": {
                "min": float(np.min(self._state.pressure.data)),
                "max": float(np.max(self._state.pressure.data))
            },
            "level_set": self._state.levelset.get_diagnostics()
        }