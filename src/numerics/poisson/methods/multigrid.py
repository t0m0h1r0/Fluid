"""
マルチグリッド法によるPoisson方程式の高速解法（JAX最適化版）
"""

from typing import Optional, Union, Tuple
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import logging

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField

logger = logging.getLogger(__name__)


class PoissonMultigridSolver(PoissonSolverBase):
    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        cycle_type: str = "V",
        num_levels: int = 3,
        omega: float = 0.8,  # 緩和パラメータ
    ):
        super().__init__(config)
        self.cycle_type = cycle_type
        self.num_levels = num_levels
        self.omega = omega
        jax.config.update("jax_enable_x64", True)
        self._setup_logging()

    def _setup_logging(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    @staticmethod
    @jax.jit
    def laplacian(u: jnp.ndarray, dx: jnp.ndarray) -> jnp.ndarray:
        """3Dラプラシアン演算子"""
        dx2 = dx**2
        return (
            (jnp.roll(u, -1, axis=0) + jnp.roll(u, 1, axis=0) - 2 * u) / dx2[0]
            + (jnp.roll(u, -1, axis=1) + jnp.roll(u, 1, axis=1) - 2 * u) / dx2[1]
            + (jnp.roll(u, -1, axis=2) + jnp.roll(u, 1, axis=2) - 2 * u) / dx2[2]
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4))
    def relax(
        u: jnp.ndarray, f: jnp.ndarray, dx: jnp.ndarray, num_iter: int, omega: float
    ) -> jnp.ndarray:
        """Weighted Jacobi法による緩和"""
        dx2 = dx**2
        inv_diag = 1.0 / (2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2]))

        def single_relax(u, _):
            # 残差の計算
            r = f - PoissonMultigridSolver.laplacian(u, dx)
            # 更新
            u_new = u + omega * inv_diag * r
            return u_new, None

        final_u, _ = lax.scan(single_relax, u, None, length=num_iter)
        return final_u

    @staticmethod
    @jax.jit
    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        """制限演算子 (Injection)"""
        return fine[::2, ::2, ::2]

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def prolong(coarse: jnp.ndarray, fine_shape: Tuple[int, ...]) -> jnp.ndarray:
        """補間演算子 (Linear)"""
        # まずコースグリッドの値を配置
        fine = jnp.zeros(fine_shape)
        fine = fine.at[::2, ::2, ::2].set(coarse)

        # x方向の補間
        fine = fine.at[1:-1:2, ::2, ::2].set(
            0.5 * (fine[:-2:2, ::2, ::2] + fine[2::2, ::2, ::2])
        )

        # y方向の補間
        fine = fine.at[:, 1:-1:2, ::2].set(
            0.5 * (fine[:, :-2:2, ::2] + fine[:, 2::2, ::2])
        )

        # z方向の補間
        fine = fine.at[:, :, 1:-1:2].set(0.5 * (fine[:, :, :-2:2] + fine[:, :, 2::2]))

        return fine

    def v_cycle(
        self, u: jnp.ndarray, f: jnp.ndarray, dx: jnp.ndarray, level: int
    ) -> jnp.ndarray:
        """Vサイクル"""

        @partial(jax.jit, static_argnums=(3,))
        def _v_cycle_impl(u, f, dx, level):
            # 最粗グリッドでの解法
            if level == self.num_levels - 1:
                return self.relax(u, f, dx * (2**level), 50, self.omega)

            # 前緩和
            u = self.relax(u, f, dx * (2**level), 2, self.omega)

            # 残差の計算
            r = f - self.laplacian(u, dx * (2**level))

            # 粗いグリッドへの制限
            r_2h = self.restrict(r)
            e_2h = jnp.zeros_like(r_2h)

            # 粗いグリッドでの補正を計算
            e_2h = _v_cycle_impl(e_2h, r_2h, dx, level + 1)

            # 補正の補間
            e_h = self.prolong(e_2h, u.shape)

            # 補正の適用
            u = u + e_h

            # 後緩和
            u = self.relax(u, f, dx * (2**level), 2, self.omega)

            return u

        return _v_cycle_impl(u, f, dx, level)

    def solve(
        self,
        rhs: Union[jnp.ndarray, ScalarField],
        initial_guess: Optional[Union[jnp.ndarray, ScalarField]] = None,
    ) -> jnp.ndarray:
        """Poisson方程式を解く"""
        logger.info("マルチグリッドソルバーを初期化中...")
        rhs_array, initial_array = self.validate_input(rhs, initial_guess)
        solution = (
            initial_array if initial_array is not None else jnp.zeros_like(rhs_array)
        )
        dx = jnp.array(self.config.dx)

        @jax.jit
        def iteration_step(state):
            u, best_u, min_residual = state
            # Vサイクルの適用
            u = self.v_cycle(u, rhs_array, dx, 0)

            # 残差計算
            r = rhs_array - self.laplacian(u, dx)
            residual = jnp.linalg.norm(r) / jnp.linalg.norm(rhs_array)

            # 最良解の更新
            best_u = jnp.where(residual < min_residual, u, best_u)
            min_residual = jnp.minimum(residual, min_residual)

            return (u, best_u, min_residual), residual

        logger.info("反復計算を開始...")
        state = (solution, solution, jnp.inf)

        for i in range(self.config.max_iterations):
            state, residual = iteration_step(state)

            if i % 10 == 0 or residual < 1e-4:
                logger.info(
                    f"反復 {i}/{self.config.max_iterations}, 残差 = {residual:.2e}"
                )

            if residual < self.config.tolerance:
                logger.info(f"収束達成: 反復 {i + 1}回, 最終残差 = {residual:.2e}")
                break

        solution, best_solution, final_residual = state
        self._iteration_count = i + 1
        self._converged = residual < self.config.tolerance
        self._error_history = [float(final_residual)]

        if not self._converged:
            logger.warning(f"最大反復回数到達: 残差 = {final_residual:.2e}")

        return best_solution
