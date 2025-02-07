"""シミュレーション用ロガーを提供するモジュール

このモジュールは、シミュレーション固有のロギング機能を提供します。
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from .config import LogConfig
from .handlers import FileLogHandler, ConsoleLogHandler, BufferedLogHandler
from .formatters import DetailedFormatter


class SimulationLogger:
    """シミュレーション用ロガークラス

    シミュレーション全体で一貫したロギングを提供し、
    進捗状況や重要なイベントを記録します。
    """

    def __init__(
        self,
        name: str,
        config: Optional[LogConfig] = None,
        parent: Optional["SimulationLogger"] = None,
    ):
        """ロガーを初期化

        Args:
            name: ロガーの名前
            config: ロギング設定
            parent: 親ロガー（階層的ロギング用）
        """
        self.name = name
        self.config = config or LogConfig()
        self.parent = parent

        # 基本設定の検証
        self.config.validate()
        self.config.create_directories()

        # ロガーの取得または作成
        self.logger = self._create_logger()

        # デバッグ情報の記録用バッファ
        self._debug_buffer = BufferedLogHandler()
        if parent is None:  # ルートロガーの場合のみ初期ログを出力
            self.logger.info(f"ロギングシステムを初期化: {name}")

    def _create_logger(self) -> logging.Logger:
        """ロガーを生成して設定

        Returns:
            設定済みのロガーインスタンス
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.level.upper()))

        # 既存のハンドラをクリア
        logger.handlers.clear()

        # ファイルハンドラの設定
        if self.config.file_logging["enabled"]:
            file_handler = FileLogHandler(
                filename=self.config.get_file_path(),
                formatter=DetailedFormatter(),
                max_bytes=self.config.file_logging["max_bytes"],
                backup_count=self.config.file_logging["backup_count"],
                level=self.config.file_logging["level"],
            )
            logger.addHandler(file_handler)

        # コンソールハンドラの設定
        if self.config.console_logging["enabled"]:
            console_handler = ConsoleLogHandler(
                level=self.config.console_logging["level"],
                use_color=self.config.console_logging["color"],
            )
            logger.addHandler(console_handler)

        # デバッグバッファの追加
        logger.addHandler(self._debug_buffer)

        return logger

    def start_section(self, name: str) -> "SimulationLogger":
        """新しいログセクションを開始

        Args:
            name: セクション名

        Returns:
            セクション用の新しいロガー
        """
        section_name = f"{self.name}.{name}"
        return SimulationLogger(section_name, self.config, self)

    def get_recent_logs(self, n: int = 100) -> list:
        """最近のログメッセージを取得

        Args:
            n: 取得するメッセージ数

        Returns:
            最近のログメッセージのリスト
        """
        return self._debug_buffer.get_logs()[-n:]

    def save_debug_info(self, path: Union[str, Path]):
        """デバッグ情報をファイルに保存

        Args:
            path: 保存先のファイルパス
        """
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for log in self._debug_buffer.get_logs():
                f.write(f"{log}\n")

    def log_error_with_context(
        self, msg: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ):
        """エラー情報をコンテキスト付きでログ出力

        Args:
            msg: エラーメッセージ
            error: 発生した例外
            context: 追加のコンテキスト情報
        """
        error_info = {
            "message": msg,
            "error_type": type(error).__name__,
            "error_msg": str(error),
            "context": context or {},
        }
        self.logger.error(f"Error occurred: {error_info}", exc_info=True)

    def log_performance(self, section: str, elapsed: float):
        """パフォーマンス情報をログ出力

        Args:
            section: 計測セクション名
            elapsed: 経過時間（秒）
        """
        self.logger.info(f"Performance - {section}: {elapsed:.3f} seconds")

    def log_simulation_state(self, state: Dict[str, Any], level: str = "info"):
        """シミュレーション状態をログ出力

        Args:
            state: 記録する状態情報
            level: ログレベル
        """
        log_func = getattr(self.logger, level.lower())
        log_func(f"Simulation State: {state}")

    def __getattr__(self, name: str):
        """未定義の属性アクセスをロガーに転送

        Args:
            name: 属性名

        Returns:
            ロガーの対応するメソッド
        """
        return getattr(self.logger, name)

    def __enter__(self):
        """コンテキストマネージャのエントリー"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャのイグジット

        エラーが発生した場合はログに記録します。
        """
        if exc_type is not None:
            self.log_error_with_context(
                "Error in simulation section", exc_val, {"section": self.name}
            )
        return False  # 例外を伝播させる
