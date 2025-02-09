"""ソルバーの基底クラスを提供するモジュール

このモジュールは、数値解法の基底となる抽象クラスを定義します。
全ての具体的なソルバーはこの基底クラスを継承します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class Solver(ABC):
    """ソルバーの基底クラス

    この抽象基底クラスは、全てのソルバーに共通のインターフェースと
    基本機能を提供します。
    """

    def __init__(
        self,
        name: str,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        logger=None,
    ):
        """ソルバーを初期化

        Args:
            name: ソルバーの名前
            tolerance: 収束判定の許容誤差
            max_iterations: 最大反復回数
            logger: ロガー（オプション）
        """
        self.name = name
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._iteration_count = 0
        self._residual_history = []
        self._start_time = None
        self._end_time = None
        self._logger = logger

    @property
    def tolerance(self) -> float:
        """収束判定の許容誤差を取得"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float):
        """収束判定の許容誤差を設定"""
        if value <= 0:
            raise ValueError("許容誤差は正の値である必要があります")
        self._tolerance = value

    @property
    def max_iterations(self) -> int:
        """最大反復回数を取得"""
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int):
        """最大反復回数を設定"""
        if value <= 0:
            raise ValueError("最大反復回数は正の整数である必要があります")
        self._max_iterations = value

    @property
    def iteration_count(self) -> int:
        """現在の反復回数を取得"""
        return self._iteration_count

    @property
    def residual_history(self) -> list:
        """残差の履歴を取得"""
        return self._residual_history.copy()

    @property
    def elapsed_time(self) -> Optional[float]:
        """計算経過時間を取得（秒）"""
        if self._start_time is None:
            return None
        end_time = self._end_time or datetime.now()
        return (end_time - self._start_time).total_seconds()

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """ソルバーの初期化"""
        pass

    @abstractmethod
    def solve(self, **kwargs) -> Dict[str, Any]:
        """ソルバーを実行"""
        pass

    def reset(self):
        """ソルバーの状態をリセット"""
        self._iteration_count = 0
        self._residual_history = []
        self._start_time = None
        self._end_time = None

    def _start_solving(self):
        """計算開始時の処理"""
        self.reset()
        self._start_time = datetime.now()
        if self._logger:
            self._logger.info(f"{self.name}ソルバーの計算を開始")

    def _end_solving(self):
        """計算終了時の処理"""
        self._end_time = datetime.now()
        if self._logger:
            self._logger.info(
                f"{self.name}ソルバーの計算を終了 (経過時間: {self.elapsed_time:.2f}秒)"
            )

    def get_status(self) -> Dict[str, Any]:
        """ソルバーの現在の状態を取得"""
        return {
            "name": self.name,
            "iteration_count": self.iteration_count,
            "residual": self._residual_history[-1] if self._residual_history else None,
            "elapsed_time": self.elapsed_time,
        }

    def __str__(self) -> str:
        """ソルバーの文字列表現を取得"""
        status = self.get_status()
        return (
            f"Solver: {status['name']}\n"
            f"Iterations: {status['iteration_count']}\n"
            f"Current Residual: {status['residual']}\n"
            f"Elapsed Time: {status['elapsed_time']:.2f}s"
        )
