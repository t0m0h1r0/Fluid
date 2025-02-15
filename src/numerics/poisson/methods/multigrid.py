"""
マルチグリッド法によるPoisson方程式の高速解法（JAX最適化版）
"""

from typing import Optional, Dict, Any, Union, Tuple
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
        jax.config.update("jax_enable_x64", True)
        self._setup_logging()

    def _setup_logging(self):
        """ロギングの設定"""
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
        """3Dラプラシアン演算子（ベクトル化版）"""
        dx2 = dx**2
        return (
            (jnp.roll(u, -1, axis=0) + jnp.roll(u, 1, axis=0) - 2 * u) / dx2[0]
            + (jnp.roll(u, -1, axis=1) + jnp.roll(u, 1, axis=1) - 2 * u) / dx2[1]
            + (jnp.roll(u, -1, axis=2) + jnp.roll(u, 1, axis=2) - 2 * u) / dx2[2]
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def red_black_relaxation(u: jnp.ndarray, f: jnp.ndarray, dx: jnp.ndarray, num_iter: int) -> jnp.ndarray:
        """Red-Black Gauss-Seidel緩和法"""
        dx2 = dx**2
        factor = 2.0 * (1/dx2[0] + 1/dx2[1] + 1/dx2[2])

        def relaxation_step(carry, _):
            u = carry
            # Red points
            red_mask = ((jnp.indices(u.shape)[0] + jnp.indices(u.shape)[1] + jnp.indices(u.shape)[2]) % 2 == 0)
            neighbor_sum = (
                (jnp.roll(u, -1, axis=0) + jnp.roll(u, 1, axis=0)) / dx2[0] +
                (jnp.roll(u, -1, axis=1) + jnp.roll(u, 1, axis=1)) / dx2[1] +
                (jnp.roll(u, -1, axis=2) + jnp.roll(u, 1, axis=2)) / dx2[2]
            )
            new_u = jnp.where(red_mask, (neighbor_sum - f) / factor, u)

            # Black points
            black_mask = ~red_mask
            neighbor_sum = (
                (jnp.roll(new_u, -1, axis=0) + jnp.roll(new_u, 1, axis=0)) / dx2[0] +
                (jnp.roll(new_u, -1, axis=1) + jnp.roll(new_u, 1, axis=1)) / dx2[1] +
                (jnp.roll(new_u, -1, axis=2) + jnp.roll(new_u, 1, axis=2)) / dx2[2]
            )
            new_u = jnp.where(black_mask, (neighbor_sum - f) / factor, new_u)
            
            return new_u, None

        final_u, _ = lax.scan(relaxation_step, u, None, length=num_iter)
        return final_u

    @staticmethod
    @jax.jit
    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        """制限演算子（フルウェイト版）"""
        # 内部点の重み付け平均
        weights = jnp.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]) / 64.0

        coarse = jnp.zeros((fine.shape[0]//2, fine.shape[1]//2, fine.shape[2]//2))
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    shifted = jnp.roll(jnp.roll(jnp.roll(
                        fine, i-1, axis=0), j-1, axis=1), k-1, axis=2)
                    coarse += weights[i,j] * shifted[::2,::2,::2]

        return coarse

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def prolong(coarse: jnp.ndarray, fine_shape: Tuple[int, ...]) -> jnp.ndarray:
        """補間演算子（三線形補間）"""
        fine = jnp.zeros(fine_shape, dtype=coarse.dtype)
        
        # コーナー点の設定
        fine = fine.at[::2, ::2, ::2].set(coarse)
        
        # エッジの補間
        fine = fine.at[1:-1:2, ::2, ::2].set(
            0.5 * (fine[:-2:2, ::2, ::2] + fine[2::2, ::2, ::2])
        )
        fine = fine.at[::2, 1:-1:2, ::2].set(
            0.5 * (fine[::2, :-2:2, ::2] + fine[::2, 2::2, ::2])
        )
        fine = fine.at[::2, ::2, 1:-1:2].set(
            0.5 * (fine[::2, ::2, :-2:2] + fine[::2, ::2, 2::2])
        )
        
        # 面の補間
        fine = fine.at[1:-1:2, 1:-1:2, ::2].set(
            0.25 * (
                fine[1:-1:2, :-2:2, ::2] + fine[1:-1:2, 2::2, ::2] +
                fine[:-2:2, 1:-1:2, ::2] + fine[2::2, 1:-1:2, ::2]
            )
        )
        
        # 内部点の補間
        fine = fine.at[1:-1:2, 1:-1:2, 1:-1:2].set(
            0.125 * (
                fine[1:-1:2, 1:-1:2, :-2:2] + fine[1:-1:2, 1:-1:2, 2::2] +
                fine[1:-1:2, :-2:2, 1:-1:2] + fine[1:-1:2, 2::2, 1:-1:2] +
                fine[:-2:2, 1:-1:2, 1:-1:2] + fine[2::2, 1:-1:2, 1:-1:2]
            )
        )

        return fine

    def v_cycle(
        self, u: jnp.ndarray, f: jnp.ndarray, dx: jnp.ndarray, level: int
    ) -> jnp.ndarray:
        """Vサイクル（最適化版）"""

        # JITコンパイルされた内部関数を定義
        @partial(jax.jit, static_argnums=(3,))
        def _v_cycle_impl(u, f, dx, level):
            # 前緩和
            u = self.red_black_relaxation(u, f, dx * (2**level), 2)
            
            if level == self.num_levels - 1:
                # 最粗レベルでの直接解法
                return self.red_black_relaxation(u, f, dx * (2**level), 50)
            
            # 残差の計算
            residual = f - self.laplacian(u, dx * (2**level))
            
            # 粗いグリッドへの制限
            coarse_residual = self.restrict(residual)
            coarse_correction = jnp.zeros_like(coarse_residual)
            
            # 粗いグリッドでの解法
            coarse_correction = _v_cycle_impl(
                coarse_correction, coarse_residual, dx, level + 1
            )
            
            # 補正の補間と適用
            correction = self.prolong(coarse_correction, u.shape)
            u = u + correction
            
            # 後緩和
            return self.red_black_relaxation(u, f, dx * (2**level), 2)

        # 内部関数を呼び出し
        return _v_cycle_impl(u, f, dx, level)

    def solve(
        self,
        rhs: Union[jnp.ndarray, ScalarField],
        initial_guess: Optional[Union[jnp.ndarray, ScalarField]] = None,
    ) -> jnp.ndarray:
        """Poisson方程式を解く"""
        logger.info("マルチグリッドソルバーを初期化中...")
        rhs_array, initial_array = self.validate_input(rhs, initial_guess)
        solution = initial_array if initial_array is not None else jnp.zeros_like(rhs_array)
        dx = jnp.array(self.config.dx)

        # JITコンパイルされた反復ステップ
        @jax.jit
        def iteration_step(state):
            """単一の反復ステップ"""
            u, best_u, min_residual = state
            u = self.v_cycle(u, rhs_array, dx, 0)
            residual = jnp.linalg.norm(rhs_array - self.laplacian(u, dx))
            best_u = jnp.where(residual < min_residual, u, best_u)
            min_residual = jnp.minimum(residual, min_residual)
            return (u, best_u, min_residual), residual

        # メインループ
        logger.info("反復計算を開始...")
        state = (solution, solution, jnp.inf)
        
        for i in range(self.config.max_iterations):
            state, residual = iteration_step(state)
            
            # 進捗表示
            if i % 10 == 0 or residual < 1e-4:
                logger.info(f"反復 {i}/{self.config.max_iterations}, 残差 = {residual:.2e}")
            
            if residual < self.config.tolerance:
                logger.info(f"収束達成: 反復 {i+1}回, 最終残差 = {residual:.2e}")
                break
            
        solution, best_solution, final_residual = state
        self._iteration_count = i + 1
        self._converged = residual < self.config.tolerance
        self._error_history = [float(final_residual)]
        
        if not self._converged:
            logger.warning(f"最大反復回数到達: 残差 = {final_residual:.2e}")
        
        return best_solution