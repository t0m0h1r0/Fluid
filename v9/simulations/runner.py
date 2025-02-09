"""シミュレーションの実行を管理するモジュール"""

import traceback
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple

from simulations.state import SimulationState
from simulations.monitor import SimulationMonitor
from physics.navier_stokes import NavierStokesSolver
from physics.levelset import LevelSetSolver


class SimulationRunner:
    """シミュレーションを実行するクラス"""

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        monitor: Optional[SimulationMonitor] = None,
    ):
        """初期化

        Args:
            config: シミュレーション設定
            logger: ロガー
            monitor: シミュレーション監視インスタンス
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # モニターの初期化
        self.monitor = monitor or SimulationMonitor(config, logger)

        # Navier-Stokesソルバーの初期化
        self.ns_solver = NavierStokesSolver(
            logger=self.logger,
            use_weno=config.get("numerical", {})
            .get("advection", {})
            .get("use_weno", True),
        )

        # Level Setソルバーの初期化
        self.ls_solver = LevelSetSolver(
            use_weno=config.get("numerical", {})
            .get("advection", {})
            .get("use_weno", True)
        )

        # ステップ管理用の変数
        self._current_state = None
        self._current_time = 0.0
        self._current_step = 0
        self._dt = None

    def initialize(self, initial_state: SimulationState):
        """シミュレーションを初期化

        Args:
            initial_state: 初期状態
        """
        # 初期状態の設定
        self._current_state = initial_state.copy()
        self._current_time = 0.0
        self._current_step = 0

        # 各フィールドの時間を初期化
        self._current_state.velocity.time = 0.0
        self._current_state.pressure.time = 0.0
        self._current_state.levelset.time = 0.0

        # モニターを初期化
        self.monitor.update(self._current_state)

    def step_forward(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """1時間ステップを進める

        Returns:
            (更新された状態, ステップ情報の辞書)のタプル
        """
        try:
            # 時間刻み幅の計算
            self._dt = self.ns_solver.compute_timestep(self._current_state.velocity)

            # Navier-Stokesソルバーで状態を更新
            advanced_state, ns_result = self.ns_solver.advance(
                self._current_state,
                dt=self._dt,
            )

            # レベルセットの移流
            ls_result = self.ls_solver.advance(
                dt=self._dt,
                phi=advanced_state.levelset,
                velocity=advanced_state.velocity,
            )

            # 時間を更新
            self._current_time += self._dt
            advanced_state.velocity.time = self._current_time
            advanced_state.pressure.time = self._current_time
            advanced_state.levelset.time = self._current_time
            self._current_step += 1

            # 現在の状態を更新
            self._current_state = advanced_state

            # モニターを更新
            self.monitor.update(self._current_state)

            # ステップ情報を統合
            step_info = {
                "dt": self._dt,
                "time": self._current_time,
                "step": self._current_step,
                "ns_result": ns_result,
                "ls_result": ls_result,
            }

            return self._current_state, step_info

        except Exception as e:
            # エラーログの出力
            self.logger.error(f"シミュレーション実行中にエラー: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def get_status(self) -> Dict[str, Any]:
        """シミュレーションの現在の状態を取得

        Returns:
            状態情報を含む辞書
        """
        return {
            "current_time": self._current_time,
            "current_step": self._current_step,
            "dt": self._dt,
        }

    def save_checkpoint(self, filepath: Path):
        """チェックポイントを保存

        Args:
            filepath: 保存先のパス
        """
        # チェックポイントデータの作成
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._current_state.save_to_file(str(filepath))

    @classmethod
    def from_checkpoint(cls, checkpoint_path, config, logger=None):
        """チェックポイントからシミュレーションを再開

        Args:
            checkpoint_path: チェックポイントファイルのパス
            config: シミュレーション設定
            logger: ロガー

        Returns:
            再開されたシミュレーションランナーと状態
        """
        # チェックポイントからの状態復元
        state = SimulationState.load_from_file(str(checkpoint_path))

        # ランナーの初期化
        monitor = SimulationMonitor(config, logger)
        runner = cls(config, logger, monitor)

        return runner, state

    def finalize(self, output_dir: Path):
        """シミュレーションの終了処理

        Args:
            output_dir: 出力ディレクトリ
        """
        # モニターのレポート生成
        self.monitor.plot_history(output_dir)
        self.monitor.generate_report(output_dir)
