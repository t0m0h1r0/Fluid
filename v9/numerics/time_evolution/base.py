from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Generic

T = TypeVar("T")


class TimeIntegrator(ABC, Generic[T]):
    """時間積分のための抽象基底クラス"""

    def __init__(self, stability_limit: float = float("inf")):
        """
        Args:
            stability_limit: 安定性限界（CFL条件など）
        """
        self._stability_limit = stability_limit

    @abstractmethod
    def integrate(self, state: T, dt: float, derivative_fn: Callable[[T], T]) -> T:
        """時間積分を実行する抽象メソッド"""
        pass

    def check_stability(self, dt: float, state_derivative: T) -> bool:
        """安定性条件をチェック"""
        return True

    @abstractmethod
    def get_order(self) -> int:
        """積分スキームの次数を返す"""
        pass

    @abstractmethod
    def get_error_estimate(self) -> float:
        """打切り誤差の推定値を返す"""
        pass
