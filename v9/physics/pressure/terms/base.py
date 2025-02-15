"""
圧力ソルバーの基底クラスと抽象インターフェース

このモジュールは、圧力方程式の解法に関する基本的な抽象化と
プロトコルを定義します。

主な数学的背景:
ポアソン方程式: ∇²p = f

抽象クラスと継承を通じて、柔軟で拡張可能な
圧力計算フレームワークを提供します。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional, Union, TypeVar

import jax.numpy as jnp

from core.field import ScalarField, VectorField


class PressureTerm(Protocol):
    """
    圧力方程式の項に対するプロトコル

    各項は以下の共通インターフェースを持つ必要があります：
    1. 名前付け
    2. 有効/無効の切り替え
    3. 項の寄与の計算
    4. 診断情報の提供
    """

    @property
    def name(self) -> str:
        """項の名前を取得"""
        ...

    @property
    def enabled(self) -> bool:
        """項が有効かどうかを取得"""
        ...

    def compute(self, solution: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        項の寄与を計算

        Args:
            solution: 現在の解
            **kwargs: 追加のパラメータ

        Returns:
            項の寄与を表す配列
        """
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


@dataclass
class PressureSolverConfig:
    """圧力ソルバーの設定を保持するデータクラス"""

    # 収束判定パラメータ
    tolerance: float = 1e-6
    max_iterations: int = 1000

    # ソルバー固有のパラメータ
    method: str = "conjugate_gradient"
    relaxation_parameter: float = 1.5  # SORなどで使用

    # 診断情報の設定
    save_residual_history: bool = True
    log_frequency: int = 10

    def validate(self):
        """設定値の妥当性を検証"""
        if not 0 < self.tolerance < 1:
            raise ValueError("許容誤差は0から1の間である必要があります")

        if self.max_iterations <= 0:
            raise ValueError("最大反復回数は正の整数である必要があります")

        valid_methods = ["conjugate_gradient", "sor", "jacobi"]
        if self.method not in valid_methods:
            raise ValueError(f"無効なソルバー方法。選択肢: {valid_methods}")


class BasePressureSolver(ABC):
    """
    ポアソン方程式ソルバーの抽象基底クラス

    主な数学的背景:
    一般化されたポアソン方程式の解法を提供:
    ∇²u = f

    継承可能な基本的な機能:
    1. 反復解法のフレームワーク
    2. 収束判定
    3. 診断情報の追跡
    """

    def __init__(
        self,
        config: Optional[PressureSolverConfig] = None,
        terms: Optional[list[PressureTerm]] = None,
    ):
        """
        圧力ソルバーを初期化

        Args:
            config: ソルバーの設定
            terms: 追加の項のリスト
        """
        self.config = config or PressureSolverConfig()
        self.terms = terms or []

        # 計算状態の追跡
        self._iteration_count = 0
        self._residual_history: list[float] = []
        self._converged = False

    @abstractmethod
    def solve(
        self, rhs: jnp.ndarray, initial_guess: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        ポアソン方程式を解く抽象メソッド

        Args:
            rhs: 右辺ベクトル
            initial_guess: 初期推定解（オプション）

        Returns:
            計算された解
        """
        pass

    def _compute_residual(
        self, solution: jnp.ndarray, rhs: jnp.ndarray, operator: callable
    ) -> float:
        """
        残差を計算

        Args:
            solution: 現在の解
            rhs: 右辺ベクトル
            operator: 離散化された演算子

        Returns:
            残差のノルム
        """
        return jnp.linalg.norm(rhs - operator(solution))

    def _apply_terms(self, solution: jnp.ndarray) -> jnp.ndarray:
        """
        追加の項を適用

        Args:
            solution: 現在の解

        Returns:
            項の寄与を合計した値
        """
        result = jnp.zeros_like(solution)
        for term in self.terms:
            if term.enabled:
                result += term.compute(solution)
        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        ソルバーの診断情報を取得

        Returns:
            診断情報を含む辞書
        """
        return {
            "iterations": self._iteration_count,
            "converged": self._converged,
            "residual_history": self._residual_history,
            "final_residual": (
                self._residual_history[-1] if self._residual_history else None
            ),
            "terms": [term.get_diagnostics() for term in self.terms],
        }


# 数値安定性のための型変数
PressureType = TypeVar("PressureType", bound=Union[ScalarField, VectorField])
