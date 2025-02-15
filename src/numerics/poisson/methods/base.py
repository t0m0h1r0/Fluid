"""
Poisson方程式ソルバーの抽象基底クラス（JAX最適化版）
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial

from ..config import PoissonSolverConfig
from core.field import ScalarField


class PoissonSolverBase(ABC):
    """Poisson方程式ソルバーの抽象基底クラス"""

    def __init__(self, config: Optional[PoissonSolverConfig] = None):
        """
        Args:
            config: ソルバーの設定パラメータ
        """
        self.config = config or PoissonSolverConfig()
        self.config.validate()

        # 状態追跡用のプロパティ
        self._iteration_count: int = 0
        self._error_history: list[float] = []
        self._converged: bool = False

        # JAXで最適化された基本演算子の初期化
        self._init_jax_operators()

    def _init_jax_operators(self):
        """JAXで最適化された基本演算子を初期化"""

        # ラプラシアン演算子をJITコンパイル
        @partial(jit, static_argnums=(0,))
        def laplacian_operator(self, u: jnp.ndarray) -> jnp.ndarray:
            dx2 = jnp.square(self.config.dx)
            return (
                (jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) - 2 * u) / dx2[0]
                + (jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) - 2 * u) / dx2[1]
                + (jnp.roll(u, 1, axis=2) + jnp.roll(u, -1, axis=2) - 2 * u) / dx2[2]
            )

        # 境界条件を考慮したバージョン
        @partial(jit, static_argnums=(0,))
        def laplacian_with_boundaries(self, u: jnp.ndarray) -> jnp.ndarray:
            # 内部点での計算
            result = self.laplacian_operator(u)

            # 境界条件の適用（デフォルトはディリクレ境界条件）
            result = result.at[0, :, :].set(0.0)
            result = result.at[-1, :, :].set(0.0)
            result = result.at[:, 0, :].set(0.0)
            result = result.at[:, -1, :].set(0.0)
            result = result.at[:, :, 0].set(0.0)
            result = result.at[:, :, -1].set(0.0)

            return result

        self.laplacian_operator = laplacian_operator
        self.laplacian_with_boundaries = laplacian_with_boundaries

    @abstractmethod
    def solve(
        self,
        rhs: Union[np.ndarray, jnp.ndarray, ScalarField],
        initial_guess: Optional[Union[np.ndarray, jnp.ndarray, ScalarField]] = None,
    ) -> jnp.ndarray:
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
        rhs: Union[np.ndarray, jnp.ndarray, ScalarField],
        initial_guess: Optional[Union[np.ndarray, jnp.ndarray, ScalarField]] = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        入力の妥当性を検証しJAX配列に変換

        Args:
            rhs: 右辺項
            initial_guess: 初期推定解（オプション）

        Returns:
            (rhs_array, initial_guess_array)のタプル。ともにJAX配列
        """
        # 右辺項の変換と検証
        if isinstance(rhs, ScalarField):
            rhs_array = jnp.array(rhs.data)
        elif isinstance(rhs, (np.ndarray, jnp.ndarray)):
            rhs_array = jnp.array(rhs)
        else:
            raise TypeError("右辺項は配列またはScalarField型である必要があります")

        # 初期推定解の変換と検証
        if initial_guess is None:
            initial_guess_array = None
        elif isinstance(initial_guess, ScalarField):
            initial_guess_array = jnp.array(initial_guess.data)
            if initial_guess_array.shape != rhs_array.shape:
                raise ValueError("初期推定解と右辺項の形状が一致しません")
        elif isinstance(initial_guess, (np.ndarray, jnp.ndarray)):
            initial_guess_array = jnp.array(initial_guess)
            if initial_guess_array.shape != rhs_array.shape:
                raise ValueError("初期推定解と右辺項の形状が一致しません")
        else:
            raise TypeError("初期推定解は配列またはScalarField型である必要があります")

        return rhs_array, initial_guess_array

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "method": self.__class__.__name__,
            "iterations": self._iteration_count,
            "converged": self._converged,
            "error_history": self._error_history,
        }
