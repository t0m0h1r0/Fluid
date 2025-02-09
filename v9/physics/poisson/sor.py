"""SOR法によるPoissonソルバーを提供するモジュール

このモジュールは、Successive Over-Relaxation (SOR)法による
Poisson方程式のソルバーを実装します。
"""

import numpy as np
from typing import Dict, Any, Optional, List
from .solver import PoissonSolver
from core.boundary import BoundaryCondition


class SORSolver(PoissonSolver):
    """SOR法によるPoissonソルバー"""

    def __init__(
        self,
        omega: float = 1.5,
        use_redblack: bool = True,
        auto_tune: bool = True,
        boundary_conditions: Optional[List[BoundaryCondition]] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        logger=None,
    ):
        """SORソルバーを初期化

        Args:
            omega: 緩和係数（1 < ω < 2）
            use_redblack: 赤黒順序付けを使用するかどうか
            auto_tune: 緩和係数を自動調整するかどうか
            boundary_conditions: 境界条件のリスト
            tolerance: 収束判定の許容誤差
            max_iterations: 最大反復回数
            logger: ロガー（オプション）
        """
        super().__init__(
            boundary_conditions=boundary_conditions,
            tolerance=tolerance,
            max_iterations=max_iterations,
            logger=logger,
        )
        self.omega = omega
        self.use_redblack = use_redblack
        self.auto_tune = auto_tune

        # 自動調整用のパラメータ
        self._spectral_radius = None
        self._update_interval = 10
        self._previous_diff = None

    def iterate(
        self, solution: np.ndarray, rhs: np.ndarray, dx: np.ndarray, **kwargs
    ) -> np.ndarray:
        """1回のSOR反復を実行"""
        result = solution.copy()

        # dx配列の正規化と設定
        if np.isscalar(dx):
            dx = np.full(result.ndim, dx)
        elif len(dx) != result.ndim:
            raise ValueError(f"dx must have length {result.ndim}, got {len(dx)}")

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
                result[mask] = (1 - self.omega) * result[
                    mask
                ] + self.omega * gauss_seidel

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
        if self.auto_tune and self.iteration_count % self._update_interval == 0:
            self._update_omega(solution, result)

        return result

    def _update_omega(self, old_solution: np.ndarray, new_solution: np.ndarray):
        """緩和係数を自動調整"""
        # 解の変化から反復行列のスペクトル半径を推定
        diff = new_solution - old_solution
        if self.iteration_count > self._update_interval:
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

    def get_status(self) -> Dict[str, Any]:
        """ソルバーの状態を取得"""
        status = super().get_diagnostics()
        status.update(
            {
                "method": "SOR",
                "omega": self.omega,
                "spectral_radius": self._spectral_radius,
                "redblack": self.use_redblack,
                "auto_tune": self.auto_tune,
            }
        )
        if self._logger:
            self._logger.debug(f"SORソルバーの状態: {status}")
        return status