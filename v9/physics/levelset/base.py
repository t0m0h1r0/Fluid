"""Level Set法の基底クラスとプロトコルを定義"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class LevelSetTerm(Protocol):
    """Level Set方程式の項のプロトコル"""

    @property
    def name(self) -> str:
        """項の名前"""
        ...

    def compute(self, levelset: Any, velocity: Any, dt: float, **kwargs) -> np.ndarray:
        """Level Set場への寄与を計算"""
        ...

    def get_diagnostics(self, levelset: Any, velocity: Any, **kwargs) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


class LevelSetTermBase:
    """Level Set項の基底実装"""

    def __init__(self, name: str):
        """基底クラスを初期化

        Args:
            name: 項の名前
        """
        self.name = name
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """項が有効かどうかを取得"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """項の有効/無効を設定"""
        self._enabled = value

    def compute_timestep(self, **kwargs) -> float:
        """項に基づく時間刻み幅の制限を計算

        デフォルトでは制限なし（float('inf')）を返す
        """
        return float("inf")

    def get_diagnostics(self, **kwargs) -> Dict[str, Any]:
        """診断情報のデフォルト実装"""
        return {"name": self.name, "enabled": self.enabled}


class LevelSetSolverBase(ABC):
    """Level Set ソルバーの基底抽象クラス"""

    def __init__(self, use_weno: bool = True, weno_order: int = 5, logger=None):
        """基底ソルバーを初期化

        Args:
            use_weno: WENOスキームを使用するかどうか
            weno_order: WENOスキームの次数
            logger: ロガー
        """
        self.use_weno = use_weno
        self.weno_order = weno_order
        self.logger = logger
        self._time = 0.0
        self._iteration_count = 0

    @abstractmethod
    def compute_derivative(self, state: Any, **kwargs) -> Any:
        """Level Set関数の時間微分を計算

        Args:
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間微分
        """
        pass

    @abstractmethod
    def compute_timestep(self, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算

        Args:
            **kwargs: 必要なパラメータ

        Returns:
            計算された時間刻み幅
        """
        pass

    def _compute_weno_reconstruction(self, values: np.ndarray, axis: int) -> np.ndarray:
        """WENOスキームによる再構築の共通実装

        Args:
            values: 再構築する値の配列
            axis: 再構築を行う軸

        Returns:
            再構築された値
        """
        # 具体的な実装は派生クラスで行う
        raise NotImplementedError("サブクラスで実装する必要があります")
