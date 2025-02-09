"""シミュレーションの実行を管理するモジュール"""

import traceback
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Tuple

from physics.navier_stokes import NavierStokesSolver
from .state import SimulationState
from .monitor import SimulationMonitor


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
        self.solver = NavierStokesSolver(
            logger=self.logger,
            use_weno=config.get("numerical", {})
            .get("advection", {})
            .get("use_weno", True),
        )

    def initialize(self, initial_state: SimulationState):
        """シミュレーションを初期化

        Args:
            initial_state: 初期状態
        """
        # 初期状態の設定
        self._current_state = initial_state.copy()

        # ソルバーを初期化
        self.solver.initialize(state=self._current_state)

        # モニターを初期化
        self.monitor.update(self._current_state)

    def step_forward(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """1時間ステップを進める

        Returns:
            (更新された状態, ステップ情報の辞書)のタプル
        """
        try:
            # Navier-Stokesソルバーで状態を更新
            state, step_info = self.solver.step_forward(self._current_state)

            # モニターを更新
            self.monitor.update(state)

            # 現在の状態を更新
            self._current_state = state

            return state, step_info

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
            "time": self.solver._total_time,
            "step": self.solver._iteration_count,
            "diagnostics": self.solver.get_diagnostics(),
        }

    def save_checkpoint(self, filepath: Path):
        """チェックポイントを保存

        Args:
            filepath: 保存先のパス
        """
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
        runner.initialize(state)

        return runner, state

    def finalize(self, output_dir: Path):
        """シミュレーションの終了処理

        Args:
            output_dir: 出力ディレクトリ
        """
        # モニターのレポート生成
        self.monitor.plot_history(output_dir)
        self.monitor.generate_report(output_dir)
