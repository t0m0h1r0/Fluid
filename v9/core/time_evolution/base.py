"""時間発展の基底クラスを提供するモジュール

このモジュールは、時間発展計算の基底となる抽象クラスを定義します。
速度場やレベルセット場など、様々な物理量の時間発展計算に共通のインターフェースを提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class TimeEvolutionBase(ABC):
    """時間発展の基底クラス"""

    def __init__(self, logger=None):
        """初期化

        Args:
            logger: ロガー
        """
        self.logger = logger
        self._time = 0.0
        self._dt = None

    @property
    def time(self) -> float:
        """現在の時刻を取得"""
        return self._time

    @property
    def dt(self) -> Optional[float]:
        """時間刻み幅を取得"""
        return self._dt

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
    def step_forward(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める

        Args:
            dt: 時間刻み幅
            **kwargs: 計算に必要なパラメータ

        Returns:
            計算された結果と診断情報
        """
        pass

    def initialize(self, **kwargs) -> None:
        """初期化処理

        Args:
            **kwargs: 初期化に必要なパラメータ
        """
        self._time = 0.0
        self._dt = None

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "time": self._time,
            "dt": self._dt,
        }
