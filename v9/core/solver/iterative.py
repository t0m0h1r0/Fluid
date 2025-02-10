"""反復法ソルバーの基底クラスを提供するモジュール

このモジュールは、反復法による方程式求解のための基底クラスを定義します。
"""

from abc import abstractmethod
from typing import Dict, Any
import numpy as np
from .base import Solver


class IterativeSolver(Solver):
    """反復法ソルバーの基底クラス

    この抽象基底クラスは、反復法による求解を行うソルバーに共通の
    機能を提供します。
    """

    def __init__(
        self, 
        name: str, 
        omega: float = 1.0, 
        **kwargs
    ):
        """反復法ソルバーを初期化

        Args:
            name: ソルバーの名前
            omega: 緩和係数
            **kwargs: 基底クラスに渡すパラメータ
        """
        super().__init__(name, **kwargs)
        self._omega = omega
        self._initial_residual = None
        self._convergence_history = []

    @property
    def omega(self) -> float:
        """緩和係数を取得"""
        return self._omega

    @omega.setter
    def omega(self, value: float):
        """緩和係数を設定

        Args:
            value: 設定する緩和係数

        Raises:
            ValueError: 不適切な値が指定された場合
        """
        if value <= 0 or value > 2:
            raise ValueError("緩和係数は0から2の間である必要があります")
        self._omega = value

    @abstractmethod
    def compute_residual(self, solution: np.ndarray, **kwargs) -> float:
        """残差を計算

        Args:
            solution: 現在の解
            **kwargs: 計算に必要なパラメータ

        Returns:
            計算された残差
        """
        pass

    @abstractmethod
    def iterate(self, solution: np.ndarray, **kwargs) -> np.ndarray:
        """1回の反復を実行

        Args:
            solution: 現在の解
            **kwargs: 計算に必要なパラメータ

        Returns:
            更新された解
        """
        pass

    def check_convergence(self, residual: float) -> bool:
        """収束判定

        Args:
            residual: 現在の残差

        Returns:
            収束したかどうか
        """
        # 初回の残差を記録
        if self._initial_residual is None:
            self._initial_residual = residual
            return False

        # 初期残差が非常に小さい場合の特別な処理
        if self._initial_residual < 1e-15:
            return residual < 1e-10

        # 相対残差による収束判定
        relative_residual = residual / self._initial_residual
        self._convergence_history.append(relative_residual)

        # 相対残差と絶対残差の両方でチェック
        return (relative_residual < self.tolerance) and (residual < 1e-10)

    def solve(self, initial_solution: np.ndarray, **kwargs) -> Dict[str, Any]:
        """反復法で方程式を解く

        Args:
            initial_solution: 初期推定解
            **kwargs: 計算に必要なパラメータ

        Returns:
            計算結果と統計情報を含む辞書

        Raises:
            RuntimeError: 最大反復回数に達しても収束しない場合
        """
        self._start_solving()

        # 初期解のコピー
        solution = initial_solution.copy()

        # 初期残差の計算
        self._initial_residual = self.compute_residual(solution, **kwargs)
        self._residual_history.append(self._initial_residual)

        while self._iteration_count < self.max_iterations:
            # 1回の反復
            solution = self.iterate(solution, **kwargs)

            # 残差の計算
            residual = self.compute_residual(solution, **kwargs)
            self._residual_history.append(residual)

            # 反復回数の更新
            self._iteration_count += 1

            # 収束判定
            if self.check_convergence(residual):
                self._end_solving()
                return {
                    "solution": solution,
                    "converged": True,
                    "iterations": self._iteration_count,
                    "residual": residual,
                    "convergence_history": self._convergence_history,
                    "elapsed_time": self.elapsed_time,
                }

        self._end_solving()
        raise RuntimeError(
            f"ソルバーが収束しませんでした: 残差 = {residual}, "
            f"相対残差 = {residual / self._initial_residual}"
        )

    def get_status(self) -> Dict[str, Any]:
        """ソルバーの現在の状態を取得"""
        status = super().get_status()
        status.update(
            {
                "omega": self._omega,
                "initial_residual": self._initial_residual,
                "current_convergence": self._convergence_history[-1]
                if self._convergence_history
                else None,
            }
        )
        return status