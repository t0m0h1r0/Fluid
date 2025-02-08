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
            laplacian += (
                np.roll(solution, 1, axis=axis)
                + np.roll(solution, -1, axis=axis)
                - 2 * solution
            ) / dx**2

        # 残差の計算と境界条件の適用
        residual = laplacian - rhs
        if self.boundary_conditions:
            for i, bc in enumerate(self.boundary_conditions):
                if bc is not None:
                    residual = bc.apply_all(residual, i)

        print(np.sqrt(np.mean(residual**2)))
        return np.sqrt(np.mean(residual**2))

    def apply_boundary_conditions(self, field: np.ndarray) -> np.ndarray:
        """境界条件を適用

        Args:
            field: 境界条件を適用する場

        Returns:
            境界条件が適用された場
        """
        result = field.copy()
        if self.boundary_conditions:
            for i, bc in enumerate(self.boundary_conditions):
                if bc is not None:
                    result = bc.apply_all(result, i)
        return result


class JacobiSolver(PoissonSolver):
    """ヤコビ法によるPoissonソルバー"""

    def iterate(
        self, solution: np.ndarray, rhs: np.ndarray, dx: float, **kwargs
    ) -> np.ndarray:
        """1回のヤコビ反復を実行

        Args:
            solution: 現在の解
            rhs: 右辺
            dx: グリッド間隔
            **kwargs: 未使用

        Returns:
            更新された解
        """
        result = np.zeros_like(solution)

        # ヤコビ反復
        for axis in range(solution.ndim):
            result += np.roll(solution, 1, axis=axis) + np.roll(solution, -1, axis=axis)

        result = (dx**2 * rhs + result) / (2 * solution.ndim)

        # 境界条件の適用
        result = self.apply_boundary_conditions(result)

        return result


class GaussSeidelSolver(PoissonSolver):
    """ガウス・ザイデル法によるPoissonソルバー

    赤黒順序付けを使用して並列化可能な実装を提供します。
    """

    def __init__(self, use_redblack: bool = True, **kwargs):
        """ガウス・ザイデルソルバーを初期化

        Args:
            use_redblack: 赤黒順序付けを使用するかどうか
            **kwargs: 基底クラスに渡すパラメータ
        """
        super().__init__(**kwargs)
        self.use_redblack = use_redblack

    def iterate(
        self, solution: np.ndarray, rhs: np.ndarray, dx: float, **kwargs
    ) -> np.ndarray:
        """1回のガウス・ザイデル反復を実行

        Args:
            solution: 現在の解
            rhs: 右辺
            dx: グリッド間隔
            **kwargs: 未使用

        Returns:
            更新された解
        """
        result = solution.copy()

        if self.use_redblack:
            # 赤黒順序付けによる更新
            for color in [0, 1]:  # 0: 赤, 1: 黒
                mask = np.zeros_like(result, dtype=bool)
                for i in range(result.ndim):
                    mask ^= np.arange(result.shape[i])[:, None, None] % 2 == color

                # 近傍点の和を計算
                neighbors_sum = np.zeros_like(result)
                for axis in range(result.ndim):
                    neighbors_sum += np.roll(result, 1, axis=axis) + np.roll(
                        result, -1, axis=axis
                    )

                # マスクされた点を更新
                result[mask] = (dx**2 * rhs[mask] + neighbors_sum[mask]) / (
                    2 * result.ndim
                )

        else:
            # 通常のガウス・ザイデル反復
            for axis in range(result.ndim):
                neighbors_sum = np.roll(result, 1, axis=axis) + np.roll(
                    result, -1, axis=axis
                )
                result = (dx**2 * rhs + neighbors_sum) / (2 * result.ndim)

        # 境界条件の適用
        result = self.apply_boundary_conditions(result)

        return result

    def get_status(self) -> Dict[str, Any]:
        """ソルバーの状態を取得"""
        status = super().get_status()
        status.update(
            {
                "method": "redblack" if self.use_redblack else "standard",
                "iteration_count": self.iteration_count,
                "residual": self.residual_history[-1]
                if self.residual_history
                else None,
            }
        )
        return status
