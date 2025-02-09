"""時間発展ソルバーの基底クラスを提供するモジュール

このモジュールは、時間発展問題を解くためのソルバーの基底クラスを定義します。
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
from .base import Solver


class TemporalSolver(Solver):
    """時間発展ソルバーの基底クラス"""

    def __init__(
        self,
        name: str,
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        logger=None,
    ):
        """時間発展ソルバーを初期化"""
        super().__init__(
            name=name, tolerance=tolerance, max_iterations=max_iterations, logger=logger
        )
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
        """CFL数を設定"""
        if value <= 0:
            raise ValueError("CFL数は正の値である必要があります")
        self._cfl = value

    @abstractmethod
    def compute_timestep(self, **kwargs) -> float:
        """時間刻み幅を計算"""
        pass

    @abstractmethod
    def advance(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップ進める"""
        pass
