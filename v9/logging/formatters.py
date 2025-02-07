"""ログフォーマッタを提供するモジュール

このモジュールは、ログメッセージの書式設定を行うフォーマッタクラスを提供します。
"""

import logging
import datetime


class DefaultFormatter(logging.Formatter):
    """標準的なログフォーマッタ

    基本的なタイムスタンプ、ログレベル、メッセージを含むフォーマットを提供します。
    """

    def __init__(self, fmt: str = None):
        """フォーマッタを初期化

        Args:
            fmt: フォーマット文字列（Noneの場合はデフォルト使用）
        """
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt)


class DetailedFormatter(logging.Formatter):
    """詳細なログフォーマッタ

    ファイル名、行番号、関数名などの詳細情報を含むフォーマットを提供します。
    """

    def __init__(self, fmt: str = None):
        """フォーマッタを初期化

        Args:
            fmt: フォーマット文字列（Noneの場合はデフォルト使用）
        """
        if fmt is None:
            fmt = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[%(filename)s:%(lineno)d] - %(message)s"
            )
        super().__init__(fmt)

    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        """時刻のフォーマット

        ミリ秒単位の精度を提供します。

        Args:
            record: ログレコード
            datefmt: 日付フォーマット文字列

        Returns:
            フォーマットされた時刻文字列
        """
        ct = datetime.datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return s


class ColoredFormatter(logging.Formatter):
    """カラー対応のログフォーマッタ

    ログレベルに応じて異なる色でメッセージを表示します。
    """

    # ANSIエスケープシーケンス
    COLORS = {
        "DEBUG": "\033[36m",  # シアン
        "INFO": "\033[32m",  # 緑
        "WARNING": "\033[33m",  # 黄
        "ERROR": "\033[31m",  # 赤
        "CRITICAL": "\033[35m",  # マゼンタ
        "RESET": "\033[0m",  # リセット
    }

    def __init__(self, fmt: str = None, use_color: bool = True):
        """フォーマッタを初期化

        Args:
            fmt: フォーマット文字列（Noneの場合はデフォルト使用）
            use_color: 色付けを使用するかどうか
        """
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """ログレコードをフォーマット

        Args:
            record: ログレコード

        Returns:
            フォーマットされたログメッセージ
        """
        if not self.use_color:
            return super().format(record)

        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)
