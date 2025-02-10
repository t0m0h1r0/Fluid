from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from numerics.poisson import PoissonConfig


class PoissonRHSCalculator:
    """
    粘性の不均一性を考慮した圧力ポアソン方程式の右辺計算クラス

    圧力方程式の右辺 ∇⋅(...) の計算を担当
    """

    def __init__(self, solver_config: PoissonConfig = None):
        """
        Args:
            solver_config: ポアソンソルバーの設定
        """
        # デフォルト設定
        self._config = solver_config or PoissonConfig()

    def compute_rhs(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        dt: float,
    ) -> np.ndarray:
        """
        粘性の不均一性を考慮した圧力ポアソン方程式の右辺を計算

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            dt: 時間刻み幅

        Returns:
            右辺項の配列
        """
        dx = velocity.dx

        # 密度と速度の発散項
        div_rho_u = self._compute_density_velocity_divergence(velocity, density, dt)

        # 粘性項の計算
        viscosity_term = self._compute_viscosity_term(velocity, viscosity)

        # 右辺を計算
        return -div_rho_u / dt + viscosity_term

    def _compute_density_velocity_divergence(
        self, velocity: VectorField, density: ScalarField, dt: float
    ) -> np.ndarray:
        """密度と速度の発散を計算"""
        dx = velocity.dx
        result = np.zeros_like(density.data)

        for i in range(velocity.ndim):
            # 密度と速度の積の勾配を計算
            rho_u = density.data * velocity.components[i].data
            div_rho_u = np.gradient(rho_u, dx, axis=i)
            result += div_rho_u

        return result

    def _compute_viscosity_term(
        self, velocity: VectorField, viscosity: ScalarField
    ) -> np.ndarray:
        """粘性の不均一性を考慮した粘性項を計算"""
        dx = velocity.dx
        result = np.zeros_like(viscosity.data)

        # 粘性勾配項の計算
        viscosity_gradient = []
        for i in range(velocity.ndim):
            viscosity_gradient.append(np.gradient(viscosity.data, dx, axis=i))

        # 粘性の不均一性を考慮した項の計算
        for i in range(velocity.ndim):
            # μ∇²u項
            laplacian_u = np.gradient(
                np.gradient(velocity.components[i].data, dx, axis=i), dx, axis=i
            )
            mu_laplacian_u = viscosity.data * laplacian_u

            # (∇μ)⋅∇u項
            grad_dot_u = sum(
                viscosity_gradient[j]
                * np.gradient(velocity.components[i].data, dx, axis=j)
                for j in range(velocity.ndim)
            )

            result += np.gradient(mu_laplacian_u, dx, axis=i) + grad_dot_u

        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {"solver_config": self._config.get_config_for_component("convergence")}
