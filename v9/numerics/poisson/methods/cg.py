"""共役勾配法（CG）によるPoissonソルバーの実装

このモジュールは、圧力ポアソン方程式を解くための共役勾配法を実装します。
共役勾配法は対称正定値な問題に対して効率的な反復解法です。
"""

import numpy as np
from typing import Optional, Dict, Any, Union
from ..base import PoissonSolverConfig
from ..solver import PoissonSolver


class ConjugateGradientSolver(PoissonSolver):
    """共役勾配法によるPoissonソルバー"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        preconditioner: str = "none",
        **kwargs,
    ):
        """
        共役勾配法ソルバーを初期化

        Args:
            config: ソルバー設定
            preconditioner: 前処理の種類 ('none', 'jacobi', 'ilu')
            **kwargs: 基底クラスに渡すパラメータ
        """
        super().__init__(config=config, **kwargs)
        self.preconditioner = preconditioner
        self._iteration_count = 0
        self._residual_history = []

    def iterate(self, solution: np.ndarray, rhs: np.ndarray, dx: float) -> np.ndarray:
        """1回の反復を実行

        Args:
            solution: 現在の解
            rhs: 右辺
            dx: グリッド間隔

        Returns:
            更新された解
        """
        if self._iteration_count == 0:
            # 初期残差とサーチ方向の計算
            self.residual = rhs - self._apply_operator(solution, dx)
            if self.preconditioner == "jacobi":
                self.z = self._apply_jacobi_preconditioner(self.residual, dx)
            else:
                self.z = self.residual.copy()
            self.p = self.z.copy()
            self.rz_old = np.sum(self.residual * self.z)

        # Ap の計算
        Ap = self._apply_operator(self.p, dx)

        # ステップサイズの計算
        pAp = np.sum(self.p * Ap)
        if abs(pAp) < 1e-14:
            raise ValueError("共役勾配法: pAp が小さすぎます")

        alpha = self.rz_old / pAp

        # 解と残差の更新
        solution += alpha * self.p
        self.residual -= alpha * Ap

        # 前処理の適用
        if self.preconditioner == "jacobi":
            self.z = self._apply_jacobi_preconditioner(self.residual, dx)
        else:
            self.z = self.residual.copy()

        # βの計算
        rz_new = np.sum(self.residual * self.z)
        beta = rz_new / self.rz_old
        self.rz_old = rz_new

        # サーチ方向の更新
        self.p = self.z + beta * self.p

        self._iteration_count += 1
        return solution

    def _apply_operator(
        self, v: np.ndarray, dx: Union[float, np.ndarray]
    ) -> np.ndarray:
        """ラプラシアン演算子を適用

        Args:
            v: 入力ベクトル
            dx: グリッド間隔（スカラーまたはベクトル）

        Returns:
            ラプラシアン演算子を適用した結果
        """
        result = np.zeros_like(v)

        # dxをベクトルとして扱う
        if np.isscalar(dx):
            dx_vec = np.full(v.ndim, dx)
        else:
            dx_vec = np.asarray(dx)

        # 各方向のラプラシアンを計算
        for axis in range(v.ndim):
            # 中心差分による2階微分
            forward = np.roll(v, -1, axis=axis)
            backward = np.roll(v, 1, axis=axis)
            result += (forward - 2 * v + backward) / (dx_vec[axis] * dx_vec[axis])

        return result

    def _apply_jacobi_preconditioner(
        self, v: np.ndarray, dx: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Jacobi前処理を適用

        Args:
            v: 入力ベクトル
            dx: グリッド間隔（スカラーまたはベクトル）

        Returns:
            前処理を適用した結果
        """
        # dxをベクトルとして扱う
        if np.isscalar(dx):
            dx_vec = np.full(v.ndim, dx)
        else:
            dx_vec = np.asarray(dx)

        # 対角項の逆数を計算（ラプラシアン演算子の場合）
        diagonal = sum(-2.0 / (dx_i * dx_i) for dx_i in dx_vec)
        return v / diagonal

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得

        Returns:
            ソルバーの診断情報
        """
        diag = super().get_diagnostics()
        diag.update(
            {
                "method": "Conjugate Gradient",
                "preconditioner": self.preconditioner,
                "iteration_count": self._iteration_count,
                "residual_history": self._residual_history,
            }
        )
        return diag
