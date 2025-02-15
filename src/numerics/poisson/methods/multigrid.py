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
    @jax.jit
    def _gauss_seidel_step(
        u: jnp.ndarray, f: jnp.ndarray, dx: jnp.ndarray
    ) -> jnp.ndarray:
        """単一のGauss-Seidelステップ（ベクトル化版）"""
        dx2 = dx**2
        temp = (
            (jnp.roll(u, -1, axis=0) + jnp.roll(u, 1, axis=0)) / dx2[0]
            + (jnp.roll(u, -1, axis=1) + jnp.roll(u, 1, axis=1)) / dx2[1]
            + (jnp.roll(u, -1, axis=2) + jnp.roll(u, 1, axis=2)) / dx2[2]
        )
        denom = 2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2])
        return u + 1.5 * (f - PoissonMultigridSolver.laplacian(u, dx)) / denom

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def relax(
        u: jnp.ndarray, f: jnp.ndarray, dx: jnp.ndarray, num_iter: int
    ) -> jnp.ndarray:
        """Gauss-Seidel緩和法（ベクトル化版）"""

        def body_fun(i, u):
            return PoissonMultigridSolver._gauss_seidel_step(u, f, dx)

        return lax.fori_loop(0, num_iter, body_fun, u)

    @staticmethod
    @jax.jit
    def restrict(fine: jnp.ndarray) -> jnp.ndarray:
        """制限演算子（ベクトル化版）"""
        return jnp.mean(
            fine.reshape(
                fine.shape[0] // 2, 2, fine.shape[1] // 2, 2, fine.shape[2] // 2, 2
            ),
            axis=(1, 3, 5),
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def prolong(coarse: jnp.ndarray, fine_shape: Tuple[int, ...]) -> jnp.ndarray:
        """補間演算子（ベクトル化版）"""
        # まず粗グリッドの点を配置
        fine = jnp.zeros(fine_shape, dtype=coarse.dtype)
        fine = fine.at[::2, ::2, ::2].set(coarse)

        # x方向の補間（内部点のみ）
        x_interp = jnp.arange(1, fine_shape[0] - 1, 2)
        fine = fine.at[x_interp, ::2, ::2].set(
            0.5 * (fine[x_interp - 1, ::2, ::2] + fine[x_interp + 1, ::2, ::2])
        )

        # y方向の補間（内部点のみ）
        y_interp = jnp.arange(1, fine_shape[1] - 1, 2)
        fine = fine.at[:, y_interp, ::2].set(
            0.5 * (fine[:, y_interp - 1, ::2] + fine[:, y_interp + 1, ::2])
        )

        # z方向の補間（内部点のみ）
        z_interp = jnp.arange(1, fine_shape[2] - 1, 2)
        fine = fine.at[:, :, z_interp].set(
            0.5 * (fine[:, :, z_interp - 1] + fine[:, :, z_interp + 1])
        )

        return fine

    def v_cycle(
        self, u: jnp.ndarray, f: jnp.ndarray, dx: jnp.ndarray, level: int
    ) -> jnp.ndarray:
        """Vサイクル（最適化版）"""

        # JITコンパイルされた内部関数を定義
        @partial(jax.jit, static_argnums=(3,))
        def _v_cycle_impl(u, f, dx, level):
            # 最粗レベルでの処理
            if level == self.num_levels - 1:
                return self.relax(u, f, dx * (2**level), 50)

            # 前緩和
            u = self.relax(u, f, dx * (2**level), 2)

            # 残差の計算
            residual = f - self.laplacian(u, dx * (2**level))

            # 粗いグリッドへの制限
            coarse_residual = self.restrict(residual)
            coarse_correction = jnp.zeros_like(coarse_residual)

            # 粗いグリッドでの解法
            coarse_correction = _v_cycle_impl(
                coarse_correction, coarse_residual, dx, level + 1
            )

            # 補正の補間
            correction = self.prolong(coarse_correction, u.shape)
            u = u + correction

            # 後緩和
            return self.relax(u, f, dx * (2**level), 2)

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
        solution = (
            initial_array if initial_array is not None else jnp.zeros_like(rhs_array)
        )
        dx = jnp.array(self.config.dx)

        # JITコンパイルされた内部関数として定義
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
            
            # 10回ごとまたはより頻繁な進捗表示
            if i % 10 == 0 or residual < 1e-4:  # 収束に近づいたら頻繁に表示
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
        
        logger.info(f"ソルバー終了: 反復回数 = {self._iteration_count}, 最終残差 = {final_residual:.2e}")
        return best_solution

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
