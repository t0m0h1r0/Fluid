"""時間発展ソルバーの基底クラスを提供するモジュール

このモジュールは、時間発展問題を解くためのソルバーの基底クラスを定義します。
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, Callable
import numpy as np
from .base import Solver

class TemporalSolver(Solver):
    """時間発展ソルバーの基底クラス
    
    この抽象基底クラスは、時間発展問題を解くソルバーに共通の
    機能を提供します。
    
    Attributes:
        time (float): 現在の時刻
        dt (float): 時間刻み幅
        cfl (float): CFL数
    """
    
    def __init__(self,
                 name: str,
                 cfl: float = 0.5,
                 min_dt: float = 1e-6,
                 max_dt: float = 1.0,
                 **kwargs):
        """時間発展ソルバーを初期化
        
        Args:
            name: ソルバーの名前
            cfl: CFL数
            min_dt: 最小時間刻み幅
            max_dt: 最大時間刻み幅
            **kwargs: 基底クラスに渡すパラメータ
        """
        super().__init__(name, **kwargs)
        self._time = 0.0
        self._dt = None
        self._cfl = cfl
        self._min_dt = min_dt
        self._max_dt = max_dt
        self._time_history = []
    
    @property
    def time(self) -> float:
        """現在の時刻を取得"""
        return self._time
    
    @property
    def dt(self) -> Optional[float]:
        """時間刻み幅を取得"""
        return self._dt
    
    @property
    def cfl(self) -> float:
        """CFL数を取得"""
        return self._cfl
    
    @cfl.setter
    def cfl(self, value: float):
        """CFL数を設定
        
        Args:
            value: 設定するCFL数
            
        Raises:
            ValueError: 負の値が指定された場合
        """
        if value <= 0:
            raise ValueError("CFL数は正の値である必要があります")
        self._cfl = value
    
    @abstractmethod
    def compute_timestep(self, **kwargs) -> float:
        """時間刻み幅を計算
        
        Args:
            **kwargs: 計算に必要なパラメータ
            
        Returns:
            計算された時間刻み幅
        """
        pass
    
    @abstractmethod
    def advance(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップ進める
        
        Args:
            dt: 時間刻み幅
            **kwargs: 計算に必要なパラメータ
            
        Returns:
            計算結果と統計情報を含む辞書
        """
        pass
    
    def solve(self, end_time: float, **kwargs) -> Dict[str, Any]:
        """指定時刻まで時間発展を計算
        
        Args:
            end_time: 計算終了時刻
            **kwargs: 計算に必要なパラメータ
            
        Returns:
            計算結果と統計情報を含む辞書
        """
        self._start_solving()
        
        results = []
        while self._time < end_time:
            # 時間刻み幅の計算
            self._dt = self.compute_timestep(**kwargs)
            self._dt = min(self._dt, end_time - self._time)
            
            # 1ステップ進める
            result = self.advance(self._dt, **kwargs)
            results.append(result)
            
            # 時刻の更新
            self._time += self._dt
            self._time_history.append(self._time)
            
            # 反復回数の更新
            self._iteration_count += 1
            
            # 残差の記録
            if 'residual' in result:
                self._residual_history.append(result['residual'])
        
        self._end_solving()
        
        return {
            'results': results,
            'time_history': self._time_history,
            'final_time': self._time,
            'iterations': self._iteration_count,
            'elapsed_time': self.elapsed_time
        }
    
    def get_status(self) -> Dict[str, Any]:
        """ソルバーの現在の状態を取得"""
        status = super().get_status()
        status.update({
            'time': self._time,
            'dt': self._dt,
            'cfl': self._cfl
        })
        return status