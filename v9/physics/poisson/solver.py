"""Poisson方程式のソルバーを提供するモジュール

このモジュールは、Poisson方程式 ∇²φ = f を解くための反復法ソルバーを実装します。
ヤコビ法とガウス・ザイデル法を提供し、境界条件との整合性を保ちます。
"""

import numpy as np
from typing import Optional, List
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
        self._converged = False  # 収束フラグ
        self._residual_history = []  # 残差の履歴

    @property
    def converged(self) -> bool:
        """収束状態を取得"""
        return self._converged

    @property
    def residual_history(self) -> List[float]:
        """残差の履歴を取得"""
        return self._residual_history.copy()

    def solve(
        self,
        initial_solution: Optional[np.ndarray] = None,
        rhs: Optional[np.ndarray] = None,
        dx: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Poissonソルバーを実行

        Args:
            initial_solution: 初期推定解
            rhs: 右辺ベクトル
            dx: グリッド間隔（スカラーまたは配列）
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

            # dx の正規化
            if np.isscalar(dx):
                dx = np.full(rhs.ndim, dx)
            elif len(dx) != rhs.ndim:
                raise ValueError(f"dxは{rhs.ndim}次元である必要があります")

            # 初期化処理
            self.initialize(**kwargs)
            self._converged = False
            self._residual_history = []

            # 反復解法の実行
            solution = initial_solution.copy()
            residual = float("inf")  # 初期残差

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
                    self._converged = True
                    return new_solution

                solution = new_solution

            # 最大反復回数に到達
            if self._logger:
                self._logger.warning(
                    f"最大反復回数に到達: 残差 = {residual:.3e}, "
                    f"相対残差 = {residual / self._residual_history[0]:.3e}"
                )
            return solution

        except Exception as e:
            # エラーハンドリング
            if self._logger:
                self._logger.warning(f"Poissonソルバーでエラー: {e}")
            raise

    def compute_residual(
        self, solution: np.ndarray, rhs: np.ndarray, dx: np.ndarray
    ) -> float:
        """残差を計算

        Args:
            solution: 現在の解
            rhs: 右辺
            dx: グリッド間隔（スカラーまたは配列）

        Returns:
            L2ノルムで測った残差
        """
        # dx の正規化
        if np.isscalar(dx):
            dx = np.full(solution.ndim, dx)
        elif len(dx) != solution.ndim:
            raise ValueError(f"dxは{solution.ndim}次元である必要があります")

        # ラプラシアンの計算
        laplacian = np.zeros_like(solution)
        for axis in range(solution.ndim):
            # 2次精度中心差分
            # 各方向のグリッド間隔を考慮
            forward = np.roll(solution, 1, axis=axis)
            backward = np.roll(solution, -1, axis=axis)
            laplacian += (forward + backward - 2 * solution) / (dx[axis] ** 2)

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
        self._converged = False
        self._residual_history = []

    def iterate(self, solution: np.ndarray, rhs: np.ndarray, dx: np.ndarray):
        """反復計算のデフォルト実装（サブクラスでオーバーライド）"""
        raise NotImplementedError("サブクラスで実装する必要があります")
