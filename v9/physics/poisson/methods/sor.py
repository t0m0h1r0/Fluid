"""SOR法によるPoissonソルバーを提供するモジュール

このモジュールは、Successive Over-Relaxation (SOR)法による
Poisson方程式のソルバーを実装します。
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union

from physics.poisson.solver import PoissonSolver
from physics.poisson.config import PoissonSolverConfig
from core.boundary import BoundaryCondition
from ..base import PoissonSolverTerm


class SORSolver(PoissonSolver):
    """SOR法によるPoissonソルバー"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        boundary_conditions: Optional[List[BoundaryCondition]] = None,
        terms: Optional[List[PoissonSolverTerm]] = None,
        **kwargs,
    ):
        """SORソルバーを初期化

        Args:
            config: ソルバー設定
            boundary_conditions: 境界条件のリスト
            terms: 追加の項
            **kwargs: 追加のパラメータ
        """
        # デフォルト設定の取得
        solver_config = config or PoissonSolverConfig()
        solver_specific = solver_config.get_config_for_component("solver_specific")

        # 緩和パラメータの取得
        self.omega = kwargs.get(
            "omega", solver_specific.get("relaxation_parameter", 1.5)
        )

        # 赤黒順序付けの設定
        self.use_redblack = kwargs.get(
            "use_redblack", solver_specific.get("use_redblack", True)
        )

        # 自動調整の設定
        self.auto_tune = kwargs.get(
            "auto_tune", solver_specific.get("auto_tune", False)
        )

        # スペクトル半径の追跡
        self._spectral_radius = None
        self._update_interval = 10
        self._previous_diff = None

        # 親クラスの初期化
        super().__init__(
            config=solver_config,
            boundary_conditions=boundary_conditions,
            terms=terms,
            **kwargs,
        )

    def initialize(self, **kwargs):
        """ソルバーを初期化

        Args:
            **kwargs: 初期化に必要なパラメータ
        """
        # 基本的な初期化処理
        self.reset()  # 基底クラスのリセットメソッド

        # オプションでより詳細な初期化が必要な場合はここに追加可能
        if self.logger:
            self.logger.info("SORソルバーを初期化")

        # スペクトル半径などの追跡変数もリセット
        self._spectral_radius = None
        self._previous_diff = None

    def iterate(
        self, solution: np.ndarray, rhs: np.ndarray, dx: Union[float, np.ndarray]
    ) -> np.ndarray:
        """1回のSOR反復を実行

        Args:
            solution: 現在の解
            rhs: 右辺ベクトル
            dx: グリッド間隔

        Returns:
            更新された解
        """
        result = solution.copy()

        # dx配列の正規化と設定
        if np.isscalar(dx):
            dx = np.full(result.ndim, dx)
        elif len(dx) != result.ndim:
            raise ValueError(f"dxは{result.ndim}次元である必要があります")

        # グリッド間隔の2乗を計算
        dx_squared = np.prod(dx) ** 2

        if self.use_redblack:
            # 赤黒順序付けによる更新
            for color in [0, 1]:  # 0: 赤, 1: 黒
                # 赤黒マスクの生成
                mask = np.ones_like(result, dtype=bool)
                for axis in range(result.ndim):
                    # 各軸で交互のインデックスを選択
                    axis_indices = np.arange(result.shape[axis]) % 2 == color
                    # ブロードキャスト可能な形に変換
                    broadcast_shape = [1] * result.ndim
                    broadcast_shape[axis] = -1
                    axis_mask = axis_indices.reshape(broadcast_shape)
                    # 論理積で絞り込み
                    mask &= axis_mask

                # 近傍点の和を計算
                neighbors_sum = np.zeros_like(result)
                for axis in range(result.ndim):
                    # 各軸で前後の点を取得
                    forward = np.roll(result, 1, axis=axis)
                    backward = np.roll(result, -1, axis=axis)
                    neighbors_sum += forward + backward

                # SOR更新
                gauss_seidel = (dx_squared * rhs[mask] + neighbors_sum[mask]) / (
                    2 * result.ndim
                )
                result[mask] = (1 - self.omega) * result[mask] + self.omega * gauss_seidel

        else:
            # 通常のSOR反復
            for axis in range(result.ndim):
                neighbors_sum = np.roll(result, 1, axis=axis) + np.roll(
                    result, -1, axis=axis
                )
                gauss_seidel = (dx_squared * rhs + neighbors_sum) / (2 * result.ndim)
                result = (1 - self.omega) * result + self.omega * gauss_seidel

        # 境界条件の適用
        if self.boundary_conditions:
            for i, bc in enumerate(self.boundary_conditions):
                if bc is not None:
                    result = bc.apply_all(result, i)

        # 必要に応じて緩和係数を調整
        if self.auto_tune and self._iteration_count % self._update_interval == 0:
            self._update_omega(solution, result)

        return result

    def _update_omega(self, old_solution: np.ndarray, new_solution: np.ndarray):
        """緩和係数を自動調整

        Args:
            old_solution: 前回の解
            new_solution: 新しい解
        """
        # 解の変化から反復行列のスペクトル半径を推定
        diff = new_solution - old_solution
        if self._iteration_count > self._update_interval:
            old_diff = self._previous_diff
            numerator = np.sum(diff * diff)
            denominator = np.sum(old_diff * old_diff)

            if denominator > 1e-10:
                new_radius = np.sqrt(numerator / denominator)

                if self._spectral_radius is None:
                    self._spectral_radius = new_radius
                else:
                    # 指数移動平均で更新
                    alpha = 0.2
                    self._spectral_radius = (
                        1 - alpha
                    ) * self._spectral_radius + alpha * new_radius

                # 最適な緩和係数を計算
                self.omega = 2 / (1 + np.sqrt(1 - self._spectral_radius**2))

        # 現在の差分を保存
        self._previous_diff = diff.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得

        Returns:
            診断情報の辞書
        """
        diag = super().get_diagnostics()
        diag.update(
            {
                "method": "SOR",
                "omega": self.omega,
                "spectral_radius": self._spectral_radius,
                "redblack_ordering": self.use_redblack,
                "auto_tune": self.auto_tune,
            }
        )
        return diag
