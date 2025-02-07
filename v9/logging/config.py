"""ロギング設定を管理するモジュール

このモジュールは、ロギングシステムの設定を管理するためのクラスを提供します。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

@dataclass
class LogConfig:
    """ロギング設定を管理するクラス
    
    Attributes:
        level: 基本ログレベル
        log_dir: ログファイル出力ディレクトリ
        file_logging: ファイルへのログ出力設定
        console_logging: コンソールへのログ出力設定
        formatters: 各ハンドラ用のフォーマット設定
    """
    level: str = "info"
    log_dir: Path = Path("logs")
    file_logging: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "filename": "simulation.log",
        "level": "info",
        "max_bytes": 10_000_000,  # 10MB
        "backup_count": 5
    })
    console_logging: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "level": "info",
        "color": True
    })
    formatters: Dict[str, str] = field(default_factory=lambda: {
        "default": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    })
    
    def __post_init__(self):
        """初期化後の処理
        
        ディレクトリパスの正規化とバリデーションを行います。
        """
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
    
    def validate(self):
        """設定の妥当性を検証
        
        Raises:
            ValueError: 無効な設定値が検出された場合
        """
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        if self.level.lower() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}")
        
        if not self.file_logging["enabled"] and not self.console_logging["enabled"]:
            raise ValueError("At least one logging handler must be enabled")
    
    def get_file_path(self, filename: Optional[str] = None) -> Path:
        """ログファイルのパスを取得
        
        Args:
            filename: 指定されたファイル名（Noneの場合はデフォルト使用）
            
        Returns:
            ログファイルの完全パス
        """
        filename = filename or self.file_logging["filename"]
        return self.log_dir / filename
    
    def create_directories(self):
        """必要なディレクトリを作成"""
        self.log_dir.mkdir(parents=True, exist_ok=True)