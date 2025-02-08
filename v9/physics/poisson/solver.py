"""Poisson方程式のソルバーを提供するモジュール

このモジュールは、Poisson方程式 ∇²φ = f を解くための反復法ソルバーを実装します。
ヤコビ法とガウス・ザイデル法を提供し、境界条件との整合性を保ちます。
"""

import numpy as np
from typing import Dict, Any, Optional, List
from core.solver import IterativeSolver
from core.boundary import BoundaryCondition


class PoissonSolver(IterativeSolver):
    """Poisson方程式の基底ソルバークラス

    このクラスは、Poisson方程式を解くための基本機能を提供します。
    具体的な反復法は派生クラスで実装します。
    """

    def __init__(
        self, boundary_conditions: Optional[List[BoundaryCondition]] = None, **kwargs
    ):
        """Poissonソルバーを初期化

        Args:
            boundary_conditions: 各方向の境界条件
            **kwargs: 基底クラスに渡すパラメータ
        """
        super().__init__(name="Poisson", **kwargs)
        self.boundary_conditions = boundary_conditions

    def solve(
        self, 
        initial_solution: Optional[np.ndarray] = None, 
        rhs: Optional[np.ndarray] = None, 
        dx: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """Poissonソルバーを実行

        Args:
            initial_solution: 初期推定解
            rhs: 右辺ベクトル
            dx: グリッド間隔
            **kwargs: 追加のパラメータ

        Returns:
            計算された解
        """
        try:
            # 初期解のセットアップ
            if initial_solution is None:
                initial_solution = np.zeros_like(rhs)
            
            # 右辺のセットアップ
            if rhs is None:
                raise ValueError("右辺ベクトルが指定されていません")

            # 初期化処理
            self.initialize(**kwargs)

            # 反復解法の実行
            solution = initial_solution.copy()

            while self._iteration_count < self.max_iterations:
                # 1回の反復
                new_solution = self.iterate(solution, rhs, dx)

                # 残差の計算
                residual = self.compute_residual(new_solution, rhs, dx)
                self._residual_history.append(residual)

                # 反復回数の更新
                self._iteration_count += 1

                # 収束判定
                if self.check_convergence(residual):
                    return new_solution

                solution = new_solution

            # 最大反復回数に到達
            return solution

        except Exception as e:
            # エラーハンドリング
            if self._logger:
                self._logger.warning(f"Poissonソルバーでエラー: {e}")
            raise

    def compute_residual(
        self, solution: np.ndarray, rhs: np.ndarray, dx: float
    ) -> float:
        """残差を計算

        Args:
            solution: 現在の解
            rhs: 右辺
            dx: グリッド間隔

        Returns:
            L2ノルムで測った残差
        """
        # ラプラシアンの計算
        laplacian = np.zeros_like(solution)
        for axis in range(solution.ndim):
            # 2次精度中心差分
            laplacian += (
                np.roll(solution, 1, axis=axis)
                + np.roll(solution, -1, axis=axis)
                - 2 * solution
            ) / dx**2

        # 残差の計算と境界条件の適用
        residual = laplacian - rhs

        # 境界条件の適用
        if self.boundary_conditions:
            for i, bc in enumerate(self.boundary_conditions):
                if bc is not None:
                    residual = bc.apply_all(residual, i)

        # L2ノルムを計算（ゼロ除算を防ぐ）
        residual_norm = np.sqrt(np.mean(residual**2))
        return max(residual_norm, 1e-15)  # 最小値を保証

    def initialize(self, **kwargs) -> None:
        """ソルバーの初期化

        Args:
            **kwargs: 初期化パラメータ
        """
        super().initialize(**kwargs)