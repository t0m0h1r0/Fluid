"""シミュレーションの実行を管理するモジュール"""

import traceback
from pathlib import Path
import logging
from typing import Dict, Any, Optional

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
        # TODO: チェックポイントからの復元処理を実装
        raise NotImplementedError("チェックポイントからの復元はまだ実装されていません")

    def _compute_timestep(self, state: SimulationState) -> float:
        """時間刻み幅を計算

        Args:
            state: 現在のシミュレーション状態

        Returns:
            計算された時間刻み幅
        """
        # 数値スキーム設定を取得
        numerical_config = self.config.get("numerical", {})

        # デフォルトの最大時間刻み幅
        max_dt = numerical_config.get("max_dt", 0.1)
        initial_dt = numerical_config.get("initial_dt", 0.001)

        # Navier-Stokesソルバーからの推奨時間刻み幅
        try:
            recommended_dt = self.ns_solver.compute_timestep(state.velocity)
        except Exception as e:
            self.logger.warning(f"時間刻み幅の計算中にエラー: {e}")
            recommended_dt = initial_dt

        # 安全のため、最大時間刻み幅を超えないようにする
        return min(recommended_dt, max_dt)

    def run(self, initial_state: SimulationState, output_dir: Path) -> SimulationState:
        """シミュレーションを実行

        Args:
            initial_state: 初期状態
            output_dir: 出力ディレクトリ

        Returns:
            最終状態
        """
        # 出力ディレクトリの準備
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # シミュレーション設定の取得
        numerical_config = self.config.get("numerical", {})
        max_time = numerical_config.get("max_time", 1.0)
        save_interval = numerical_config.get("save_interval", 0.1)

        try:
            # 初期状態の設定
            current_state = initial_state.copy()
            current_time = 0.0

            # シミュレーションループ
            next_save_time = save_interval
            while current_time < max_time:
                # 時間刻み幅の計算
                dt = self._compute_timestep(current_state)

                # Navier-Stokesソルバーで速度場と圧力場を更新
                velocity = current_state.velocity
                pressure = current_state.pressure
                levelset = current_state.levelset
                properties = current_state.properties

                # レベルセットの移流
                ls_result = self.ls_solver.advance(dt, phi=levelset, velocity=velocity)

                # Navier-Stokesソルバーで状態を更新
                ns_result = self.ns_solver.advance(
                    state=current_state, dt=dt, properties=properties
                )

                # 時間を更新
                current_time += dt
                current_state.velocity.time = current_time
                current_state.pressure.time = current_time
                current_state.levelset.time = current_time

                # モニターを更新（引数を1つに）
                self.monitor.update(current_state)

                # 結果を保存（必要に応じて）
                if current_time >= next_save_time:
                    # TODO: 結果の保存処理を実装
                    next_save_time += save_interval

            # 最終的な結果を生成
            self.monitor.plot_history(output_dir)
            self.monitor.generate_report(output_dir)

            return current_state

        except Exception as e:
            # エラーログの出力
            self.logger.error(f"シミュレーション実行中にエラーが発生: {e}")
            self.logger.error(traceback.format_exc())

            # エラー時の後処理
            try:
                # モニターに現在の状態を記録
                self.monitor.update(current_state)
                self.monitor.plot_history(output_dir)
                self.monitor.generate_report(output_dir)
            except Exception as log_error:
                self.logger.error(f"エラー後の後処理中に例外が発生: {log_error}")

            # 例外を再送出
            raise
