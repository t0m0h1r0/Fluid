"""共役勾配法（CG）によるPoissonソルバーの実装

このモジュールは、圧力ポアソン方程式を解くための共役勾配法を実装します。
共役勾配法は対称正定値な問題に対して効率的な反復解法です。
"""

import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
from ..base import PoissonSolverBase, PoissonSolverConfig
from ..solver import PoissonSolver


class ConjugateGradientSolver(PoissonSolver):
    """共役勾配法によるPoissonソルバー"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        preconditioner: str = "none",
        **kwargs
    ):
        """共役勾配法ソルバーを初期化

        Args:
            config: ソルバー設定
            preconditioner: 前処理の種類 ('none', 'jacobi', 'ilu')
            **kwargs: 基底クラスに渡すパラメータ
        """
        super().__init__(config=config, **kwargs)
        self.preconditioner = preconditioner
        self._iteration_count = 0
        self._residual_history = []
        self._initial_residual_norm = None

    def solve(
        self, rhs: np.ndarray, initial_solution: Optional[np.ndarray] = None, dx: Union[float, np.ndarray] = 1.0
    ) -> np.ndarray:
        """Poisson方程式を解く

        Args:
            rhs: 右辺ベクトル
            initial_solution: 初期推定解（オプション）
            dx: グリッド間隔

        Returns:
            計算された解
        """
        # 初期化
        if initial_solution is None:
            solution = np.zeros_like(rhs)
        else:
            solution = initial_solution.copy()

        # 初期残差の計算
        self.residual = rhs - self._apply_operator(solution, dx)
        residual_norm = np.linalg.norm(self.residual)
        
        # ゼロ右辺のチェック
        rhs_norm = np.linalg.norm(rhs)
        if rhs_norm < 1e-15:
            return np.zeros_like(rhs)

        self._initial_residual_norm = residual_norm
        self._residual_history = [residual_norm]

        # 前処理の適用
        if self.preconditioner == "jacobi":
            self.z = self._apply_jacobi_preconditioner(self.residual, dx)
        else:
            self.z = self.residual.copy()

        # 初期サーチ方向
        self.p = self.z.copy()
        self.rz_old = np.sum(self.residual * self.z)

        # メインの反復ループ
        for i in range(self.max_iterations):
            # Ap の計算
            Ap = self._apply_operator(self.p, dx)
            pAp = np.sum(self.p * Ap)

            # ステップサイズの計算
            if abs(pAp) < 1e-14:
                # サーチ方向が非常に小さい場合
                if residual_norm < self.tolerance * rhs_norm:
                    # 既に十分収束している
                    break
                else:
                    # 新しいサーチ方向で再開
                    self.p = self.residual.copy()
                    Ap = self._apply_operator(self.p, dx)
                    pAp = np.sum(self.p * Ap)
                    if abs(pAp) < 1e-14:
                        raise ValueError("共役勾配法: 適切なサーチ方向が見つかりません")

            alpha = self.rz_old / pAp

            # 解と残差の更新
            solution += alpha * self.p
            self.residual -= alpha * Ap

            # 収束判定
            residual_norm = np.linalg.norm(self.residual)
            self._residual_history.append(residual_norm)
            
            if residual_norm < self.tolerance * rhs_norm:
                break

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

        self._iteration_count = i + 1
        return solution

    def _apply_operator(self, v: np.ndarray, dx: Union[float, np.ndarray]) -> np.ndarray:
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

    def _apply_jacobi_preconditioner(self, v: np.ndarray, dx: Union[float, np.ndarray]) -> np.ndarray:
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
        return v / (diagonal + 1e-14)  # ゼロ除算防止

    def get_convergence_info(self) -> Dict[str, Any]:
        """収束情報を取得"""
        return {
            "iterations": self._iteration_count,
            "initial_residual": self._initial_residual_norm,
            "final_residual": self._residual_history[-1] if self._residual_history else None,
            "residual_history": self._residual_history,
            "converged": self._iteration_count < self.max_iterations
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update({
            "method": "Conjugate Gradient",
            "preconditioner": self.preconditioner,
            "convergence_info": self.get_convergence_info()
        })
        return diag