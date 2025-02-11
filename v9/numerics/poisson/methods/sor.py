"""
SOR法によるPoissonソルバーの実装

SOR（Successive Over-Relaxation）法は、
Gauss-Seidel法を緩和パラメータ（omega）で加速する反復法です。
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union

from numerics.poisson import PoissonSolver
from core.boundary import BoundaryCondition
from ..base import PoissonSolverTerm
from ..config import PoissonSolverConfig


class SORSolver(PoissonSolver):
    """SOR法によるPoissonソルバー"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        boundary_conditions: Optional[List[BoundaryCondition]] = None,
        terms: Optional[List[PoissonSolverTerm]] = None,
        **kwargs,
    ):
        """
        SORソルバーを初期化

        Args:
            config: ソルバー設定
            boundary_conditions: 境界条件のリスト
            terms: 追加の項
            **kwargs: 追加のパラメータ
        """
        # デフォルト設定の取得
        solver_config = config or PoissonSolverConfig()

        # デフォルトのSOR関連パラメータ
        solver_specific = solver_config.get_config_for_component("solver_specific")
        
        # 緩和係数の取得（デフォルト1.5）
        self.omega = kwargs.get(
            "omega", solver_specific.get("relaxation_parameter", 1.5)
        )

        # 親クラスの初期化
        super().__init__(
            config=solver_config,
            boundary_conditions=boundary_conditions,
            terms=terms,
            **kwargs,
        )

    def iterate(
        self, solution: np.ndarray, rhs: np.ndarray, dx: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        SOR法による1回の反復計算

        Args:
            solution: 現在の解
            rhs: 右辺
            dx: グリッド間隔

        Returns:
            更新された解
        """
        # dx の正規化
        if np.isscalar(dx):
            dx = np.full(solution.ndim, dx)
        elif len(dx) != solution.ndim:
            raise ValueError(f"dxは{solution.ndim}次元である必要があります")

        # 計算結果のコピー
        result = solution.copy()

        # 各次元についてSOR更新
        for axis in range(solution.ndim):
            # 各軸方向の近傍点からの寄与を計算
            neighbors_sum = (
                np.roll(result, 1, axis=axis) + 
                np.roll(result, -1, axis=axis)
            )

            # SOR更新の計算
            correction = (
                rhs + (neighbors_sum) / (2 * solution.ndim)
            ) / (2 / dx[axis]**2 + 1e-10)

            # 緩和パラメータの適用
            result = (1 - self.omega) * result + self.omega * correction

        # 境界条件の適用
        if self.boundary_conditions:
            for i, bc in enumerate(self.boundary_conditions):
                if bc is not None:
                    result = bc.apply_all(result, i)

        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        診断情報を取得

        Returns:
            診断情報の辞書
        """
        diag = super().get_diagnostics()
        diag.update({
            "method": "SOR",
            "omega": self.omega,
        })
        return diag