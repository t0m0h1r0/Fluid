"""
JAX共役勾配法によるPoisson方程式の数値解法

理論的背景:
- 対称正定値行列に対する高効率な反復解法
- 自動微分と追跡可能な計算グラフを活用
- JAXの最適化機能による高速な数値計算

数学的定式化:
∇²u = f の離散化された形式を解く
"""

from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from core.field import ScalarField
from .base import PoissonSolverBase, PoissonSolverConfig


class PoissonCGSolver(PoissonSolverBase):
    """
    JAX共役勾配法によるPoissonソルバー

    対称正定値な線形システムに対する高速な反復解法
    """

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        preconditioner: Optional[str] = None,
    ):
        """
        JAX共役勾配法ソルバーを初期化

        Args:
            config: ソルバー設定
            preconditioner: 前処理の種類 ('jacobi', None)
        """
        super().__init__(config or PoissonSolverConfig())
        self.preconditioner = preconditioner

    @partial(jit, static_argnums=(0,))
    def _apply_laplacian(self, solution: ScalarField) -> ScalarField:
        """
        ラプラシアン演算子を適用

        Args:
            solution: 解候補のスカラー場

        Returns:
            ラプラシアンを適用した結果
        """
        # 各軸方向の2階微分の和を計算
        result = ScalarField(solution.shape, solution.dx)

        for axis in range(solution.ndim):
            # 中心差分による2階微分
            grad1 = jnp.gradient(solution.data, solution.dx[axis], axis=axis)
            grad2 = jnp.gradient(grad1, solution.dx[axis], axis=axis)
            result.data += grad2

        return result

    @partial(jit, static_argnums=(0,))
    def _inner_product(self, field1: ScalarField, field2: ScalarField) -> jnp.ndarray:
        """
        スカラー場の内積を計算

        Args:
            field1: 第1のスカラー場
            field2: 第2のスカラー場

        Returns:
            内積値
        """
        return jnp.sum(field1.data * field2.data)

    @partial(jit, static_argnums=(0,))
    def _jacobi_precondition(self, residual: ScalarField) -> ScalarField:
        """
        ヤコビ前処理を適用

        Args:
            residual: 残差場

        Returns:
            前処理を適用した残差場
        """
        # ラプラシアン演算子の対角成分の逆数を計算
        diagonal_inv = ScalarField(residual.shape, residual.dx)
        for axis in range(residual.ndim):
            diagonal_inv.data += 1.0 / (residual.dx[axis] ** 2)

        # 前処理された残差
        preconditioned = ScalarField(residual.shape, residual.dx)
        preconditioned.data = residual.data / diagonal_inv.data

        return preconditioned

    @partial(jit, static_argnums=(0,))
    def solve(
        self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None
    ) -> ScalarField:
        """
        Poisson方程式を共役勾配法で解く

        Args:
            rhs: 右辺項 f
            initial_guess: 初期推定解（省略時はゼロベクトル）

        Returns:
            解 u
        """
        # 入力の妥当性検証
        self.validate_input(rhs, initial_guess)

        # 初期解の準備
        solution = (
            initial_guess.copy()
            if initial_guess is not None
            else ScalarField(rhs.shape, rhs.dx, initial_value=0.0)
        )

        def cg_step(state):
            """共役勾配法の1ステップを定義"""
            solution, residual, direction, rz_old = state

            # Aの行列ベクトル積（ラプラシアン）
            A_direction = self._apply_laplacian(
                ScalarField(direction.shape, direction.dx, direction.data)
            )

            # ステップサイズの計算
            alpha_numerator = rz_old
            alpha_denominator = self._inner_product(direction, A_direction)
            alpha = alpha_numerator / (alpha_denominator + 1e-10)

            # 解と残差の更新
            new_solution = ScalarField(
                solution.shape, solution.dx, solution.data + alpha * direction.data
            )
            new_residual = ScalarField(
                residual.shape, residual.dx, residual.data - alpha * A_direction.data
            )

            # 前処理
            if self.preconditioner == "jacobi":
                preconditioned_residual = self._jacobi_precondition(new_residual)
            else:
                preconditioned_residual = new_residual

            # 内積の計算
            rz_new = self._inner_product(new_residual, preconditioned_residual)

            # 共役方向の更新
            beta = rz_new / (rz_old + 1e-10)
            new_direction = ScalarField(
                preconditioned_residual.shape,
                preconditioned_residual.dx,
                preconditioned_residual.data + beta * direction.data,
            )

            return (new_solution, new_residual, new_direction, rz_new)

        # 初期状態の準備
        initial_residual = rhs - self._apply_laplacian(solution)

        if self.preconditioner == "jacobi":
            preconditioned_residual = self._jacobi_precondition(initial_residual)
        else:
            preconditioned_residual = initial_residual

        initial_direction = preconditioned_residual
        initial_rz = self._inner_product(initial_residual, preconditioned_residual)

        initial_state = (solution, initial_residual, initial_direction, initial_rz)

        # メイン反復
        def cg_loop(i, state):
            """共役勾配法の収束判定を含むループ"""
            solution, residual, direction, rz_old = state

            # 残差のノルムを計算
            residual_norm = jnp.linalg.norm(residual.data)
            max_norm = jnp.linalg.norm(rhs.data)

            # 収束判定
            is_converged = residual_norm <= self.config.tolerance * max_norm

            # 最大反復回数による制限
            is_max_iter = i >= self.config.max_iterations

            # 条件分岐
            new_state = jax.lax.cond(
                is_converged | is_max_iter,
                lambda _: state,
                lambda _: cg_step(state),
                operand=None,
            )

            return new_state

        # 反復計算の実行
        final_state = jax.lax.fori_loop(
            0, self.config.max_iterations, cg_loop, initial_state
        )

        # 最終的な解を返却
        return final_state[0]

    def compute_residual(self, solution: ScalarField, rhs: ScalarField) -> float:
        """
        残差を計算

        Args:
            solution: 解
            rhs: 右辺項

        Returns:
            残差のノルム
        """
        # ラプラシアンと残差の計算
        residual = rhs - self._apply_laplacian(solution)

        # 残差のノルムを返却
        return float(jnp.linalg.norm(residual.data))

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        ソルバーの診断情報を取得

        Returns:
            診断情報の辞書
        """
        # 親クラスの診断情報を取得し拡張
        diag = super().get_diagnostics()
        diag.update(
            {
                "preconditioner": self.preconditioner,
            }
        )
        return diag
