"""シミュレーション全体を管理するモジュール

このモジュールは、シミュレーションの全体的な管理と
各コンポーネント間の連携を担当します。
"""

from pathlib import Path
from typing import Dict, Any, Optional
from logger import SimulationLogger
from .initializer import SimulationInitializer
from .runner import SimulationRunner
from .monitor import SimulationMonitor


class SimulationManager:
    """シミュレーション管理クラス

    シミュレーション全体の実行フローを管理し、
    各コンポーネント間の連携を取ります。
    """

    def __init__(self, config: Dict[str, Any], logger: SimulationLogger):
        """シミュレーションマネージャーを初期化

        Args:
            config: シミュレーション設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger.start_section("manager")

        # 出力ディレクトリの設定
        self.output_dir = Path(config["visualization"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 各コンポーネントの初期化
        self.initializer = SimulationInitializer(config, logger)
        self.monitor = SimulationMonitor(config, logger)
        self.runner = SimulationRunner(config, logger, self.monitor)

    def run_simulation(self, checkpoint_path: Optional[Path] = None) -> int:
        """シミュレーションを実行

        Args:
            checkpoint_path: チェックポイントファイルのパス（省略可）

        Returns:
            終了コード（0: 成功, 1: 失敗）
        """
        try:
            if checkpoint_path:
                # チェックポイントから再開
                self.logger.info(f"チェックポイントから再開: {checkpoint_path}")
                runner, state = SimulationRunner.from_checkpoint(
                    checkpoint_path, self.config, self.logger
                )
            else:
                # 新規シミュレーション
                self.logger.info("新規シミュレーションを開始")
                state = self.initializer.create_initial_state()

            # シミュレーション実行
            final_state = self.runner.run(state, self.output_dir)

            # 結果の解析とレポート生成
            self.monitor.plot_history(self.output_dir)
            self.monitor.generate_report(self.output_dir)

            summary = self.monitor.get_summary()
            self.logger.info("シミュレーション正常終了")
            self.logger.info(f"サマリー: {summary}")

            return 0

        except Exception as e:
            self.logger.log_error_with_context(
                "シミュレーション実行中にエラーが発生", e
            )
            return 1

    def cleanup(self):
        """リソースの解放とクリーンアップを実行"""
        try:
            self.monitor.save_statistics()
            self.logger.info("クリーンアップ完了")
        except Exception as e:
            self.logger.warning(f"クリーンアップ中にエラー: {e}")
