"""ログハンドラを提供するモジュール

このモジュールは、ログの出力先を管理するハンドラクラスを提供します。
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from .formatters import DefaultFormatter, ColoredFormatter


class FileLogHandler(logging.handlers.RotatingFileHandler):
    """ファイルログハンドラ

    ログをファイルに出力し、ファイルのローテーションを管理します。
    """

    def __init__(
        self,
        filename: Path,
        formatter: Optional[logging.Formatter] = None,
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
        encoding: str = "utf-8",
        level: str = "INFO",
    ):
        """ハンドラを初期化

        Args:
            filename: ログファイルのパス
            formatter: ログフォーマッタ
            max_bytes: 1ファイルの最大サイズ（バイト）
            backup_count: 保持する過去ログの数
            encoding: ファイルのエンコーディング
            level: ログレベル
        """
        super().__init__(
            filename=str(filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
        )

        self.setFormatter(formatter or DefaultFormatter())
        self.setLevel(getattr(logging, level.upper()))


class ConsoleLogHandler(logging.StreamHandler):
    """コンソールログハンドラ

    ログを標準出力に出力します。
    """

    def __init__(
        self,
        formatter: Optional[logging.Formatter] = None,
        level: str = "INFO",
        use_color: bool = True,
    ):
        """ハンドラを初期化

        Args:
            formatter: ログフォーマッタ
            level: ログレベル
            use_color: 色付きログを使用するかどうか
        """
        super().__init__()

        if use_color:
            self.setFormatter(formatter or ColoredFormatter(use_color=True))
        else:
            self.setFormatter(formatter or DefaultFormatter())

        self.setLevel(getattr(logging, level.upper()))


class BufferedLogHandler(logging.Handler):
    """バッファ付きログハンドラ

    ログメッセージをメモリ上にバッファリングし、必要に応じて一括出力します。
    デバッグやテスト時に便利です。
    """

    def __init__(self, capacity: int = 1000):
        """ハンドラを初期化

        Args:
            capacity: バッファの最大容量
        """
        super().__init__()
        self.capacity = capacity
        self.buffer = []

    def emit(self, record: logging.LogRecord):
        """ログレコードをバッファに追加

        Args:
            record: ログレコード
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(self.format(record))

    def get_logs(self) -> list:
        """バッファ内のログを取得"""
        return self.buffer.copy()

    def clear(self):
        """バッファをクリア"""
        self.buffer.clear()


class MultiProcessLogHandler(logging.handlers.SocketHandler):
    """マルチプロセス対応ログハンドラ

    複数のプロセスからのログを安全に処理します。
    """

    def __init__(self, host: str = "localhost", port: int = 9020):
        """ハンドラを初期化

        Args:
            host: ログサーバーのホスト
            port: ログサーバーのポート
        """
        super().__init__(host, port)
        self.setFormatter(DefaultFormatter())

    def emit(self, record: logging.LogRecord):
        """ログレコードを送信

        Args:
            record: ログレコード
        """
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)
