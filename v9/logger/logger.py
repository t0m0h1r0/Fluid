import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from .config import LogConfig
from .handlers import FileLogHandler, ConsoleLogHandler, BufferedLogHandler
from .formatters import DetailedFormatter


class SimulationLogger:
    """シミュレーション用ロガークラス"""

    def __init__(
        self,
        name: str,
        config: Optional[LogConfig] = None,
        parent: Optional["SimulationLogger"] = None,
    ):
        """ロガーを初期化"""
        self.name = name
        self.config = config or LogConfig()
        self.parent = parent

        # 基本設定の検証
        self.config.validate()
        self.config.create_directories()

        # Pythonの標準ロガーを作成
        self._logger = self._create_logger()

        # デバッグ情報の記録用バッファ
        self._debug_buffer = BufferedLogHandler()
        self._logger.addHandler(self._debug_buffer)

        if parent is None:  # ルートロガーの場合のみ初期ログを出力
            self.info(f"ロギングシステムを初期化: {name}")

    def _create_logger(self) -> logging.Logger:
        """ロガーを生成して設定"""
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

        return logger

    # 基本的なロギングメソッドを直接実装
    def debug(self, msg: str, *args, **kwargs):
        """デバッグレベルのログを出力"""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """情報レベルのログを出力"""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """警告レベルのログを出力"""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """エラーレベルのログを出力"""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """クリティカルレベルのログを出力"""
        self._logger.critical(msg, *args, **kwargs)

    def start_section(self, name: str) -> "SimulationLogger":
        """新しいログセクションを開始"""
        section_name = f"{self.name}.{name}"
        return SimulationLogger(section_name, self.config, self)

    def get_recent_logs(self, n: int = 100) -> list:
        """最近のログメッセージを取得"""
        return self._debug_buffer.get_logs()[-n:]

    def save_debug_info(self, path: Union[str, Path]):
        """デバッグ情報をファイルに保存"""
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for log in self._debug_buffer.get_logs():
                f.write(f"{log}\n")

    def log_error_with_context(
        self, msg: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ):
        """エラー情報をコンテキスト付きでログ出力"""
        error_info = {
            "message": msg,
            "error_type": type(error).__name__,
            "error_msg": str(error),
            "context": context or {},
        }
        self._logger.error(f"Error occurred: {error_info}", exc_info=True)

    def log_performance(self, section: str, elapsed: float):
        """パフォーマンス情報をログ出力"""
        self._logger.info(f"Performance - {section}: {elapsed:.3f} seconds")

    def log_simulation_state(self, state: Dict[str, Any], level: str = "info"):
        """シミュレーション状態をログ出力"""
        log_func = getattr(self._logger, level.lower())
        log_func(f"Simulation State: {state}")

    def __enter__(self):
        """コンテキストマネージャのエントリー"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャのイグジット"""
        if exc_type is not None:
            self.log_error_with_context(
                "Error in simulation section", exc_val, {"section": self.name}
            )
        return False  # 例外を伝播させる
