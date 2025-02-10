"""時間発展の基底クラスを提供するモジュール

このモジュールは、時間発展計算の基底となる抽象クラスを定義します。
速度場やレベルセット場など、様々な物理量の時間発展計算に共通のインターフェースを提供します。
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, Protocol
from datetime import datetime


class TimeEvolutionTerm(Protocol):
    """時間発展の項のプロトコル"""

    @property
    def name(self) -> str:
        """項の名前"""
        ...

    def compute(self, state: Any, dt: float, **kwargs) -> Any:
        """時間微分を計算"""
        ...

    def get_diagnostics(self, state: Any, **kwargs) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


class TimeEvolutionConfig(Protocol):
    """時間発展設定のプロトコル"""

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        ...

    def get_config_for_component(self, component: str) -> Dict[str, Any]:
        """特定のコンポーネントの設定を取得"""
        ...


class TimeEvolutionBase:
    """時間発展の基底クラス"""

    def __init__(self, config: Optional[TimeEvolutionConfig] = None, logger=None):
        """初期化

        Args:
            config: 時間発展設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger

        # 時間管理
        self._time = 0.0
        self._dt = None

        # 計算履歴
        self._time_history: list = []
        self._iteration_count = 0
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    @property
    def time(self) -> float:
        """現在の時刻を取得"""
        return self._time

    @property
    def dt(self) -> Optional[float]:
        """時間刻み幅を取得"""
        return self._dt

    @property
    def iteration_count(self) -> int:
        """現在の反復回数を取得"""
        return self._iteration_count

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
    def compute_derivative(self, state: Any, **kwargs) -> Any:
        """時間微分を計算

        Args:
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間微分
        """
        pass

    def initialize(self, state: Any = None, **kwargs) -> None:
        """初期化処理

        Args:
            state: 初期状態（オプション）
            **kwargs: 初期化に必要なパラメータ
        """
        # 時間と反復回数のリセット
        self._time = 0.0
        self._dt = None
        self._iteration_count = 0
        self._time_history = []

        # 計算開始時刻の記録
        self._start_time = datetime.now()
        self._end_time = None

        # ロギング
        if self.logger:
            self.logger.info("時間発展計算を初期化")

    def finalize(self):
        """計算終了処理"""
        # 計算終了時刻の記録
        self._end_time = datetime.now()

        # ロギング
        if self.logger:
            elapsed_time = (self._end_time - self._start_time).total_seconds()
            self.logger.info(
                f"時間発展計算を終了 (経過時間: {elapsed_time:.2f}秒, "
                f"反復回数: {self._iteration_count})"
            )

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得

        Returns:
            診断情報の辞書
        """
        diagnostics = {
            "current_time": self._time,
            "iteration_count": self._iteration_count,
            "last_timestep": self._dt,
            "time_history": self._time_history,
        }

        # 経過時間の追加
        if self._start_time:
            end_time = self._end_time or datetime.now()
            diagnostics["elapsed_time"] = (end_time - self._start_time).total_seconds()

        return diagnostics
