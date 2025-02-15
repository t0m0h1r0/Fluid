"""
Poisson方程式ソルバーのための抽象基底クラス

このモジュールは、Poisson方程式 ∇²u = f を解くための
抽象基底クラスを提供します。

数値偏微分方程式における基本的な考え方:
1. 離散化された楕円型偏微分方程式の解法
2. 反復法による近似解の探索
3. 収束判定と誤差評価

主な抽象メソッド:
- solve(): 方程式を解く
- compute_residual(): 残差を計算
- validate_input(): 入力の妥当性を検証
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Protocol, runtime_checkable

from core.field import ScalarField


@runtime_checkable
class BoundaryConditionProtocol(Protocol):
    """境界条件のプロトコル"""

    def apply(self, field: ScalarField) -> ScalarField:
        """境界条件を適用"""
        ...


@dataclass
class PoissonSolverConfig:
    """Poisson方程式ソルバーの設定"""

    max_iterations: int = 1000
    tolerance: float = 1e-6
    relaxation_parameter: float = 1.0  # SOR法などで使用
    boundary_conditions: List[Optional[BoundaryConditionProtocol]] = field(
        default_factory=list
    )


class PoissonSolverBase(ABC):
    """
    Poisson方程式ソルバーの抽象基底クラス

    支配方程式: ∇²u = f

    数値的アプローチの基本設計:
    1. 離散化スキーム（中心差分など）の選択
    2. 反復法による解の近似
    3. 収束判定と誤差評価
    """

    def __init__(self, config: Optional[PoissonSolverConfig] = None):
        """
        ソルバーを初期化

        Args:
            config: ソルバーの設定パラメータ
        """
        self.config = config or PoissonSolverConfig()

        # 計算状態の追跡
        self._iteration_count = 0
        self._residual_history: List[float] = []
        self._converged = False

    @abstractmethod
    def solve(
        self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None
    ) -> ScalarField:
        """
        Poisson方程式を解く抽象メソッド

        Args:
            rhs: 右辺項 f
            initial_guess: 初期推定解（オプション）

        Returns:
            解 u
        """
        pass

    @abstractmethod
    def compute_residual(self, solution: ScalarField, rhs: ScalarField) -> float:
        """
        残差を計算する抽象メソッド

        Args:
            solution: 現在の解
            rhs: 右辺項

        Returns:
            残差のノルム
        """
        pass

    def validate_input(
        self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None
    ) -> None:
        """
        入力の妥当性を検証

        Args:
            rhs: 右辺項
            initial_guess: 初期推定解（オプション）

        Raises:
            ValueError: 入力が不正な場合
        """
        if not isinstance(rhs, ScalarField):
            raise TypeError("右辺項はScalarField型である必要があります")

        if initial_guess is not None:
            if not isinstance(initial_guess, ScalarField):
                raise TypeError("初期推定解はScalarField型である必要があります")

            if initial_guess.shape != rhs.shape:
                raise ValueError("初期推定解と右辺項の形状が一致しません")

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        ソルバーの診断情報を取得

        Returns:
            診断情報の辞書
        """
        return {
            "iteration_count": self._iteration_count,
            "converged": self._converged,
            "final_residual": (
                self._residual_history[-1] if self._residual_history else None
            ),
            "residual_history": self._residual_history,
            "config": {
                "max_iterations": self.config.max_iterations,
                "tolerance": self.config.tolerance,
                "relaxation_parameter": self.config.relaxation_parameter,
            },
        }

    def reset(self) -> None:
        """計算状態をリセット"""
        self._iteration_count = 0
        self._residual_history = []
        self._converged = False
