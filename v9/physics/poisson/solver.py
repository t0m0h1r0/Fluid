"""Poisson方程式のソルバーを提供するモジュール

このモジュールは、Poisson方程式 ∇²φ = f を解くための反復法ソルバーを実装します。
"""

import numpy as np
from typing import Optional, List, Dict, Any, Union

from core.solver import IterativeSolver
from core.boundary import BoundaryCondition
from .base import PoissonSolverBase, PoissonSolverConfig, PoissonSolverTerm
from .config import PoissonSolverConfig


class PoissonSolver(PoissonSolverBase, IterativeSolver):
    """Poisson方程式の基底ソルバークラス

    このクラスは、Poisson方程式を解くための基本機能を提供します。
    具体的な反復法は派生クラスで実装します。
    """

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        boundary_conditions: Optional[List[BoundaryCondition]] = None,
        terms: Optional[List[PoissonSolverTerm]] = None,
        **kwargs,
    ):
        """Poissonソルバーを初期化

        Args:
            config: ソルバー設定
            boundary_conditions: 各方向の境界条件
            terms: 追加の項
            **kwargs: 基底クラスに渡すパラメータ
        """
        # デフォルト設定の処理
        config = config or PoissonSolverConfig()
        kwargs.setdefault("name", "Poisson")
        kwargs.setdefault("tolerance", config.convergence.get("tolerance", 1e-6))
        kwargs.setdefault(
            "max_iterations", config.convergence.get("max_iterations", 1000)
        )

        # 基底クラスの初期化
        PoissonSolverBase.__init__(
            self,
            config=config,
            boundary_conditions=boundary_conditions,
            logger=kwargs.get("logger"),
        )
        IterativeSolver.__init__(self, **kwargs)

        # 追加の項
        self.terms = terms or []

        # 収束判定フラグ
        self._converged = False

    def solve(
        self,
        rhs: np.ndarray,
        initial_solution: Optional[np.ndarray] = None,
        dx: Union[float, np.ndarray] = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Poisson方程式を解く

        Args:
            rhs: 右辺ベクトル
            initial_solution: 初期推定解
            dx: グリッド間隔
            **kwargs: 追加のパラメータ

        Returns:
            計算された解
        """
        try:
            # 初期解のセットアップ
            if initial_solution is None:
                initial_solution = np.zeros_like(rhs)

            # dx の正規化
            if np.isscalar(dx):
                dx = np.full(rhs.ndim, dx)
            elif len(dx) != rhs.ndim:
                raise ValueError(f"dxは{rhs.ndim}次元である必要があります")

            # 初期化処理
            self.reset()
            self._converged = False

            # 反復解法の実行
            solution = initial_solution.copy()
            residual = float("inf")  # 初期残差

            diagnostics_config = self.config.get_config_for_component("diagnostics")
            log_frequency = diagnostics_config.get("log_frequency", 10)

            while self._iteration_count < self.max_iterations:
                # 1回の反復
                new_solution = self.iterate(solution, rhs, dx)

                # 残差の計算
                residual = self.compute_residual(new_solution, rhs, dx)
                self._residual_history.append(residual)

                # 反復回数の更新
                self._iteration_count += 1

                # ログ出力
                if self._iteration_count % log_frequency == 0 and self.logger:
                    self.logger.info(
                        f"反復 {self._iteration_count}: 残差 = {residual:.3e}"
                    )

                # 収束判定
                if self.check_convergence(residual):
                    self._converged = True
                    break

                solution = new_solution

            # 結果の後処理
            if not self._converged and self.logger:
                self.logger.warning(
                    f"最大反復回数に到達: 残差 = {residual:.3e}, "
                    f"相対残差 = {residual / self._residual_history[0]:.3e}"
                )

            return solution

        except Exception as e:
            if self.logger:
                self.logger.error(f"Poissonソルバー実行中にエラー: {e}")
            raise

    def compute_residual(
        self, solution: np.ndarray, rhs: np.ndarray, dx: Union[float, np.ndarray]
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

    def iterate(
        self, solution: np.ndarray, rhs: np.ndarray, dx: Union[float, np.ndarray]
    ) -> np.ndarray:
        """デフォルトの反復計算（サブクラスでオーバーライド）

        Args:
            solution: 現在の解
            rhs: 右辺
            dx: グリッド間隔

        Raises:
            NotImplementedError: サブクラスで実装する必要がある
        """
        raise NotImplementedError("具体的な反復法はサブクラスで実装する必要があります")

    def check_convergence(self, residual: float) -> bool:
        """収束判定

        Args:
            residual: 現在の残差

        Returns:
            収束したかどうか
        """
        # 収束判定設定の取得
        convergence_config = self.config.get_config_for_component("convergence")
        relative_tolerance = convergence_config.get("relative_tolerance", False)

        # 初期残差の記録（最初の呼び出し時）
        if not self._residual_history:
            return False

        # 絶対残差または相対残差による収束判定
        if relative_tolerance:
            # 相対残差
            return (
                residual / self._residual_history[0] < self.tolerance
                and residual < self.tolerance
            )
        else:
            # 絶対残差
            return residual < self.tolerance

    def get_status(self) -> Dict[str, Any]:
        """ソルバーの状態を取得

        Returns:
            ソルバーの状態を表す辞書
        """
        status = super().get_status()
        status.update(
            {
                "terms_count": len(self.terms),
                "boundary_conditions_count": len(self.boundary_conditions),
                "config": {
                    "convergence": self.config.get_config_for_component("convergence"),
                    "solver_specific": self.config.get_config_for_component(
                        "solver_specific"
                    ),
                },
            }
        )
        return status
