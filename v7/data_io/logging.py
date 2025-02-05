import logging
import sys
from pathlib import Path
from typing import Optional, Union

def setup_logging(
    level: int = logging.INFO, 
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    ロギングの設定を行う関数

    Args:
        level (int): ロギングレベル
        log_file (Optional[Union[str, Path]]): ログファイルのパス
        log_format (str): ログのフォーマット文字列
    """
    # ルートロガーの設定
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[]  # 既存のハンドラをクリア
    )
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # ルートロガーにコンソールハンドラを追加
    logging.getLogger().addHandler(console_handler)
    
    # ファイルハンドラ（オプション）
    if log_file:
        # ログファイルのディレクトリを作成
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイルハンドラ
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        
        # ルートロガーにファイルハンドラを追加
        logging.getLogger().addHandler(file_handler)

def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    名前付きロガーを取得する関数

    Args:
        name (Optional[str]): ロガー名
        level (int): ロギングレベル

    Returns:
        logging.Logger: 設定されたロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger