"""
マルチグリッド法によるPoisson方程式の高速解法

理論的背景:
- 異なる解像度のグリッドを用いて高速に収束
- 誤差の異なるスケールを効率的に抑制
- 計算量を大幅に削減

主要な実装戦略:
1. 粗いグリッドと細かいグリッドの交互使用
2. 平滑化（Smoothing）と制限（Restriction）
3. 補間（Prolongation）による誤差修正
"""

from typing import Optional

import jax.numpy as jnp
from jax import jit
from functools import partial

from core.field import ScalarField
from .base import PoissonSolverBase, PoissonSolverConfig


class PoissonMultigridSolver(PoissonSolverBase):
    """
    マルチグリッド法によるPoissonソルバー

    高速かつ高精度な数値解法を提供
    """

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        cycle_type: str = "V",
        num_levels: int = 3,
    ):
        """
        マルチグリッドソルバーを初期化

        Args:
            config: ソルバー設定
            cycle_type: マルチグリッドサイクルの種類 ('V', 'W')
            num_levels: グリッドの階層数
        """
        super().__init__(config or PoissonSolverConfig())
        self.cycle_type = cycle_type
        self.num_levels = num_levels

    @partial(jit, static_argnums=(0,))
    def _restriction(self, field: ScalarField) -> ScalarField:
        """
        グリッドを粗くする（制限演算子）

        Args:
            field: 入力スカラー場

        Returns:
            粗いグリッドのスカラー場
        """
        # 2倍の粗さのグリッドを生成
        restricted_shape = tuple(max(1, s // 2) for s in field.shape)
        restricted_dx = tuple(2.0 * d for d in field.dx)

        restricted_data = jnp.zeros(restricted_shape)

        # バイリニア補間による制限
        for i in range(restricted_shape[0]):
            for j in range(restricted_shape[1]):
                for k in range(restricted_shape[2]):
                    slice_x = slice(2 * i, 2 * i + 2)
                    slice_y = slice(2 * j, 2 * j + 2)
                    slice_z = slice(2 * k, 2 * k + 2)

                    local_patch = field.data[slice_x, slice_y, slice_z]
                    restricted_data = restricted_data.at[i, j, k].set(
                        jnp.mean(local_patch)
                    )

        return ScalarField(
            restricted_shape, restricted_dx, initial_value=restricted_data
        )

    @partial(jit, static_argnums=(0,))
    def _prolongation(self, field: ScalarField) -> ScalarField:
        """
        グリッドを細かくする（補間演算子）

        Args:
            field: 入力スカラー場

        Returns:
            細かいグリッドのスカラー場
        """
        # 2倍の細かさのグリッドを生成
        prolonged_shape = tuple(s * 2 for s in field.shape)
        prolonged_dx = tuple(d / 2 for d in field.dx)

        prolonged_data = jnp.zeros(prolonged_shape)

        # バイリニア補間による拡大
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                for k in range(field.shape[2]):
                    prolonged_data = prolonged_data.at[2 * i, 2 * j, 2 * k].set(
                        field.data[i, j, k]
                    )

                    # 補間点の追加
                    if (
                        i + 1 < field.shape[0]
                        and j + 1 < field.shape[1]
                        and k + 1 < field.shape[2]
                    ):
                        prolonged_data = prolonged_data.at[
                            2 * i + 1, 2 * j + 1, 2 * k + 1
                        ].set(
                            0.125
                            * (
                                field.data[i, j, k]
                                + field.data[i + 1, j, k]
                                + field.data[i, j + 1, k]
                                + field.data[i, j, k + 1]
                                + field.data[i + 1, j + 1, k]
                                + field.data[i + 1, j, k + 1]
                                + field.data[i, j + 1, k + 1]
                                + field.data[i + 1, j + 1, k + 1]
                            )
                        )

        return ScalarField(prolonged_shape, prolonged_dx, initial_value=prolonged_data)

    @partial(jit, static_argnums=(0,))
    def _smooth(
        self, solution: ScalarField, rhs: ScalarField, num_smoothing_steps: int = 3
    ) -> ScalarField:
        """
        ガウス・ザイデル平滑化

        Args:
            solution: 現在の解
            rhs: 右辺項
            num_smoothing_steps: 平滑化のステップ数

        Returns:
            平滑化された解
        """

        def single_smooth_step(x):
            """単一の平滑化ステップ"""
            smoothed = x.copy()

            for axis in range(x.ndim):
                # 中心差分による近似
                laplacian = jnp.zeros_like(x.data)
                for offset in [-1, 1]:
                    shifted = jnp.roll(x.data, offset, axis=axis)
                    laplacian += (shifted - x.data) / (x.dx[axis] ** 2)

                # SOR法による更新
                omega = 1.5  # 緩和パラメータ
                smoothed.data += omega * (rhs.data - laplacian) / (2 * x.ndim)

            return smoothed

        # 指定された回数の平滑化を実行
        for _ in range(num_smoothing_steps):
            solution = single_smooth_step(solution)

        return solution

    @partial(jit, static_argnums=(0,))
    def _compute_residual(self, solution: ScalarField, rhs: ScalarField) -> ScalarField:
        """
        残差を計算

        Args:
            solution: 現在の解
            rhs: 右辺項

        Returns:
            残差場
        """
        # ラプラシアンの計算
        laplacian = ScalarField(solution.shape, solution.dx)

        for axis in range(solution.ndim):
            # 2階微分の計算
            grad1 = jnp.gradient(solution.data, solution.dx[axis], axis=axis)
            grad2 = jnp.gradient(grad1, solution.dx[axis], axis=axis)
            laplacian.data += grad2

        # 残差の計算
        residual = ScalarField(rhs.shape, rhs.dx)
        residual.data = rhs.data - laplacian.data

        return residual

    def solve(
        self, rhs: ScalarField, initial_guess: Optional[ScalarField] = None
    ) -> ScalarField:
        """
        マルチグリッド法でPoisson方程式を解く

        Args:
            rhs: 右辺項
            initial_guess: 初期推定解

        Returns:
            解
        """
        # 入力の妥当性検証
        self.validate_input(rhs, initial_guess)

        # 初期解の準備
        solution = (
            initial_guess.copy()
            if initial_guess is not None
            else ScalarField(rhs.shape, rhs.dx, initial_value=0.0)
        )

        # 各階層のグリッドを準備
        grids = [solution]
        rhs_levels = [rhs]

        # グリッド階層の生成
        for _ in range(self.num_levels - 1):
            grids.append(self._restriction(grids[-1]))
            rhs_levels.append(self._restriction(rhs_levels[-1]))

        def multigrid_cycle(
            solution: ScalarField, rhs: ScalarField, level: int
        ) -> ScalarField:
            """
            マルチグリッドサイクル

            Args:
                solution: 現在の解
                rhs: 右辺項
                level: 現在のグリッドレベル

            Returns:
                修正された解
            """
            # ベースケース（最も粗いグリッド）
            if level == self.num_levels - 1:
                # 直接法で解く（最も粗いグリッドなので）
                return self._solve_coarse_grid(solution, rhs)

            # 前平滑化
            solution = self._smooth(solution, rhs, num_smoothing_steps=3)

            # 残差の計算
            residual = self._compute_residual(solution, rhs)

            # 残差を粗いグリッドに制限
            coarse_residual = self._restriction(residual)

            # 粗いグリッドで誤差を計算
            coarse_solution = ScalarField(
                coarse_residual.shape,
                coarse_residual.dx,
                initial_value=jnp.zeros_like(coarse_residual.data),
            )

            # 再帰的に誤差を修正
            error = multigrid_cycle(coarse_solution, coarse_residual, level + 1)

            # 補間によって誤差を拡大
            prolonged_error = self._prolongation(error)

            # 解の修正
            solution.data += prolonged_error.data

            # 後平滑化
            solution = self._smooth(solution, rhs, num_smoothing_steps=3)

            return solution

        # サイクルタイプに応じた実行
        for _ in range(self.config.max_iterations):
            solution = multigrid_cycle(solution, rhs, 0)

            # 収束判定
            residual_norm = jnp.linalg.norm(self._compute_residual(solution, rhs).data)
            rhs_norm = jnp.linalg.norm(rhs.data)

            if residual_norm <= self.config.tolerance * rhs_norm:
                break

        return solution

    def _solve_coarse_grid(
        self, solution: ScalarField, rhs: ScalarField
    ) -> ScalarField:
        """
        最も粗いグリッドで直接解を計算

        Args:
            solution: 初期解
            rhs: 右辺項

        Returns:
            解
        """
        # 最も粗いグリッドなので直接法で解く
        # 簡略化のため、NumPyの線形代数ソルバーを使用
        # 実際の実装では、グリッドサイズに応じた適切な方法を選択
        grid_matrix = jnp.zeros((solution.shape[0], solution.shape[0]))

        # 簡易的な離散化（より精密な実装が必要）
        for i in range(solution.shape[0]):
            grid_matrix = grid_matrix.at[i, i].set(1.0)
            if i > 0:
                grid_matrix = grid_matrix.at[i, i - 1].set(-0.5)
            if i < solution.shape[0] - 1:
                grid_matrix = grid_matrix.at[i, i + 1].set(-0.5)

        # 線形方程式を解く
        coeff_solution = jnp.linalg.solve(grid_matrix, rhs.data.reshape(-1))

        return ScalarField(
            solution.shape,
            solution.dx,
            initial_value=coeff_solution.reshape(solution.shape),
        )

    def compute_residual(self, solution: ScalarField, rhs: ScalarField) -> float:
        """
        残差のノルムを計算

        Args:
            solution: 解
            rhs: 右辺項

        Returns:
            残差のノルム
        """
        residual = self._compute_residual(solution, rhs)
        return float(jnp.linalg.norm(residual.data))
