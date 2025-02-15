"""
時間発展ソルバーの抽象基底クラス

数値時間発展計算のための抽象インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
import jax.numpy as jnp

from core.field import ScalarField


@dataclass
class PoissonSolverConfig:
    """Poisson方程式ソルバーの設定"""

    max_iterations: int = 1000
    tolerance: float = 1e-6
    relaxation_parameter: float = 1.0
    dx: Optional[np.ndarray] = None


class PoissonSolverBase(ABC):
    """
    Poisson方程式ソルバーの抽象基底クラス

    汎用的な時間積分のためのテンプレートメソッドパターンを提供
    """

    def __init__(self, config: Optional[PoissonSolverConfig] = None):
        """
        時間積分器を初期化

        Args:
            config: 時間積分の設定パラメータ
        """
        # 設定パラメータの検証
        self._validate_config(config or PoissonSolverConfig())

        # 設定の保存
        self.config = config or PoissonSolverConfig()

        # 状態追跡用のプロパティ
        self._iteration_count: int = 0
        self._error_history: List[float] = []
        self._converged: bool = False

    def _validate_config(self, config: PoissonSolverConfig):
        """
        設定パラメータの妥当性を検証

        Args:
            config: 検証する設定パラメータ

        Raises:
            ValueError: 不正な設定が見つかった場合
        """
        if config.max_iterations <= 0:
            raise ValueError("最大反復回数は正の整数である必要があります")

        if config.tolerance <= 0:
            raise ValueError("収束判定の許容誤差は正の値である必要があります")

        if not 0 < config.relaxation_parameter <= 2:
            raise ValueError("緩和パラメータは0から2の間である必要があります")

    @abstractmethod
    def solve(
        self,
        rhs: np.ndarray | jnp.ndarray | ScalarField,
        initial_guess: Optional[np.ndarray | jnp.ndarray | ScalarField] = None,
    ) -> np.ndarray | jnp.ndarray:
        """
        Poisson方程式を解く抽象メソッド

        Args:
            rhs: 右辺項
            initial_guess: 初期推定解（オプション）

        Returns:
            解
        """
        pass

    def validate_input(
        self,
        rhs: np.ndarray | jnp.ndarray | ScalarField,
        initial_guess: Optional[np.ndarray | jnp.ndarray | ScalarField] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        入力の妥当性を検証

        Args:
            rhs: 右辺項
            initial_guess: 初期推定解（オプション）

        Returns:
            (rhs_array, initial_guess_array)のタプル。ともにnumpy配列
        """
        # 右辺項の変換と検証
        if isinstance(rhs, ScalarField):
            rhs_array = np.array(rhs.data)
        elif isinstance(rhs, (np.ndarray, jnp.ndarray)):
            rhs_array = np.array(rhs)
        else:
            raise TypeError("右辺項は配列またはScalarField型である必要があります")

        # 初期推定解の変換と検証
        if initial_guess is None:
            initial_guess_array = None
        elif isinstance(initial_guess, ScalarField):
            initial_guess_array = np.array(initial_guess.data)
            if initial_guess_array.shape != rhs_array.shape:
                raise ValueError("初期推定解と右辺項の形状が一致しません")
        elif isinstance(initial_guess, (np.ndarray, jnp.ndarray)):
            initial_guess_array = np.array(initial_guess)
            if initial_guess_array.shape != rhs_array.shape:
                raise ValueError("初期推定解と右辺項の形状が一致しません")
        else:
            raise TypeError("初期推定解は配列またはScalarField型である必要があります")

        return rhs_array, initial_guess_array

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        ソルバーの診断情報を取得

        Returns:
            診断情報の辞書
        """
        return {
            "method": self.__class__.__name__,
            "iterations": self._iteration_count,
            "converged": self._converged,
            "error_history": self._error_history,
        }
