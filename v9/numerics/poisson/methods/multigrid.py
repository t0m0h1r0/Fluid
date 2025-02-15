"""
マルチグリッド法によるPoisson方程式の高速解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union, Tuple
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField


class PoissonMultigridSolver(PoissonSolverBase):
    """マルチグリッド法によるPoissonソルバー（JAX最適化版）"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        cycle_type: str = "V",
        num_levels: int = 3,
    ):
        """
        Args:
            config: ソルバーの設定
            cycle_type: マルチグリッドサイクルの種類 ('V', 'W')
            num_levels: グリッドの階層数
        """
        super().__init__(config)
        self.cycle_type = cycle_type
        self.num_levels = num_levels

        # JAXの最適化設定
        self._jax_config()
        self._init_multigrid_operators()

    def _jax_config(self):
        """JAXの最適化設定"""
        # デバイス間の計算を可能にする
        jax.config.update("jax_platform_name", "cpu")

        # 32ビット浮動小数点数を使用（メモリ効率と精度のバランス）
        jax.config.update("jax_enable_x64", False)

    def _init_multigrid_operators(self):
        """マルチグリッド法に特化したJAX最適化演算子を初期化"""

        @partial(jit, static_argnums=(1,))
        def restrict(
            fine_grid: jnp.ndarray, shape: Tuple[int, int, int]
        ) -> jnp.ndarray:
            """制限演算子（JAX最適化版）"""
            coarse_shape = tuple(s // 2 for s in shape)

            # ベクトル化された平均計算
            def block_mean(block):
                return jnp.mean(block)

            # 3D配列のブロック平均を計算
            vmap_block_mean = vmap(vmap(vmap(block_mean)))

            # 2x2x2のブロックで平均を計算
            result = vmap_block_mean(
                fine_grid.reshape(
                    coarse_shape[0], 2, coarse_shape[1], 2, coarse_shape[2], 2
                )
                .transpose(0, 2, 4, 1, 3, 5)
                .reshape(coarse_shape[0], coarse_shape[1], coarse_shape[2], 8)
            )

            return result

        @partial(jit, static_argnums=(1,))
        def prolongate(
            coarse_grid: jnp.ndarray, fine_shape: Tuple[int, int, int]
        ) -> jnp.ndarray:
            """補間演算子（JAX最適化版）"""
            fine_grid = jnp.zeros(fine_shape)

            # 直接点の補間（高速なインデックス代入）
            fine_grid = fine_grid.at[::2, ::2, ::2].set(coarse_grid)

            # エッジの補間（ベクトル化された線形補間）
            fine_grid = fine_grid.at[1::2, ::2, ::2].set(
                (fine_grid[:-1:2, ::2, ::2] + fine_grid[2::2, ::2, ::2]) / 2
            )
            fine_grid = fine_grid.at[::2, 1::2, ::2].set(
                (fine_grid[::2, :-1:2, ::2] + fine_grid[::2, 2::2, ::2]) / 2
            )
            fine_grid = fine_grid.at[::2, ::2, 1::2].set(
                (fine_grid[::2, ::2, :-1:2] + fine_grid[::2, ::2, 2::2]) / 2
            )

            return fine_grid

        @partial(jit, static_argnums=(3,))
        def smooth(
            solution: jnp.ndarray,
            rhs: jnp.ndarray,
            dx: jnp.ndarray,
            is_coarse_grid: bool = False,
        ) -> jnp.ndarray:
            """スムーサー（JAX最適化版）"""
            dx2 = dx * dx
            num_iterations = 50 if is_coarse_grid else 2

            def update_single_point(i, j, k, u):
                """単一点の更新を行う内部関数"""
                neighbors_sum = (
                    (u[i + 1, j, k] + u[i - 1, j, k]) / dx2[0]
                    + (u[i, j + 1, k] + u[i, j - 1, k]) / dx2[1]
                    + (u[i, j, k + 1] + u[i, j, k - 1]) / dx2[2]
                )
                coeff = 2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2])

                return (1 - 1.5) * u[i, j, k] + (1.5 / coeff) * (
                    neighbors_sum - rhs[i, j, k]
                )

            # SOR法による更新を反復
            for _ in range(num_iterations):
                # ベクトル化された更新
                for i in range(1, solution.shape[0] - 1):
                    for j in range(1, solution.shape[1] - 1):
                        for k in range(1, solution.shape[2] - 1):
                            solution = solution.at[i, j, k].set(
                                update_single_point(i, j, k, solution)
                            )

            return solution

        # オペレータの設定
        self.restrict = restrict
        self.prolongate = prolongate
        self.smooth = smooth

        # 追加の最適化：ラプラシアンのJIT最適化
        self.laplacian_operator = jit(self.laplacian_operator)

    def solve(
        self,
        rhs: Union[jnp.ndarray, ScalarField],
        initial_guess: Optional[Union[jnp.ndarray, ScalarField]] = None,
    ) -> jnp.ndarray:
        """Poisson方程式を解く

        Args:
            rhs: 右辺項
            initial_guess: 初期推定解（オプション）

        Returns:
            解
        """
        # 入力の検証と配列への変換
        rhs_array, initial_array = self.validate_input(rhs, initial_guess)

        # 初期解の準備
        solution = (
            initial_array if initial_array is not None else jnp.zeros_like(rhs_array)
        )

        # グリッド間隔
        dx = jnp.array(self.config.dx)

        @jit
        def multigrid_cycle(solution, rhs, level):
            """単一のマルチグリッドサイクル（JIT最適化）"""
            # 最も粗いレベルでの解法
            if level == self.num_levels - 1:
                return self.smooth(solution, rhs, dx, is_coarse_grid=True)

            # 前平滑化
            solution = self.smooth(solution, rhs, dx, is_coarse_grid=False)

            # 残差の計算
            residual = rhs - self.laplacian_operator(solution)

            # 粗いグリッドへの制限
            coarse_residual = self.restrict(residual, residual.shape)
            coarse_solution = jnp.zeros_like(coarse_residual)

            # 粗いグリッドでの解法
            coarse_solution = multigrid_cycle(
                coarse_solution, coarse_residual, level + 1
            )

            # 補正の補間と適用
            correction = self.prolongate(coarse_solution, solution.shape)
            solution = solution + correction

            # 後平滑化
            solution = self.smooth(solution, rhs, dx, is_coarse_grid=False)

            return solution

        # マルチグリッドサイクルの反復
        solution_history = [solution]
        residual_history = []

        for iteration in range(self.config.max_iterations):
            # マルチグリッドサイクルの適用
            solution = multigrid_cycle(solution, rhs_array, 0)
            solution_history.append(solution)

            # 残差の計算
            residual = jnp.max(jnp.abs(rhs_array - self.laplacian_operator(solution)))
            residual_history.append(float(residual))

            # 収束判定
            if residual < self.config.tolerance:
                self._converged = True
                break

        # 最終的な状態の更新
        self._iteration_count = iteration + 1
        self._error_history = residual_history

        return solution

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "solver_type": "Multigrid",
                "cycle_type": self.cycle_type,
                "num_levels": self.num_levels,
                "final_residual": self._error_history[-1]
                if self._error_history
                else None,
            }
        )
        return diag
