from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

@dataclass
class SimulationState:
    """シミュレーションの状態を保持するクラス"""
    
    # 時間に関する状態
    current_step: int = 0
    current_time: float = 0.0
    dt: float = 0.0
    last_save_time: float = 0.0
    
    # 実行状態
    is_running: bool = False
    is_paused: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # フィールドデータ
    fields: Dict[str, Any] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.fields is None:
            self.fields = {}
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def update_time(self, dt: float):
        """時間の更新"""
        self.current_time += dt
        self.current_step += 1
        self.dt = dt
    
    def should_save(self, save_interval: float) -> bool:
        """保存が必要かどうかを判定"""
        if self.dt == 0:
            return False
        
        time_since_last_save = self.current_time - self.last_save_time
        return time_since_last_save >= save_interval
    
    def mark_save(self):
        """保存時刻の記録"""
        self.last_save_time = self.current_time
    
    def get_elapsed_time(self) -> float:
        """経過時間を取得（秒）"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    def finish(self):
        """シミュレーション終了時の処理"""
        self.is_running = False
        self.end_time = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return {
            'step': self.current_step,
            'time': self.current_time,
            'dt': self.dt,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'elapsed_time': self.get_elapsed_time()
        }
