"""
二相流のナビエ・ストークス（NS）方程式と圧力ポアソン方程式の導出

理論的背景:
1. 基本方程式
連続の式（質量保存則）:
∂ρ/∂t + ∇⋅(ρu) = 0

ナビエ・ストークス方程式（運動方程式）:
∂(ρu)/∂t + ∇⋅(ρu⊗u) = -∇p + ∇⋅τ + ρg + Fσ

2. 圧力ポアソン方程式の導出
方程式の両辺に発散∇⋅を作用させると：
∇⋅(∂(ρu)/∂t + ∇⋅(ρu⊗u)) = -∇²p + ∇⋅(∇⋅τ) + ∇⋅(ρg) + ∇⋅Fσ

粘性の不均一性を考慮した最終的な形:
∇²p = ∇⋅(∂(ρu)/∂t + ∇⋅(ρu⊗u)) - ∇⋅(ρg) - ∇⋅Fσ
       + ∇⋅((∇μ)⋅∇u) - ∇⋅(∇⋅[μ(∇u + (∇u)T)])

特に二相流では:
- 密度と粘性の不連続性を考慮
- 界面での物理量の補間が重要
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from numerics.poisson import PoissonSolver, PoissonConfig


class PressureProjectionSolver:
    """
    圧力投影法による速度場の発散除去

    二相流のナビエ・ストークス方程式を解くための圧力投影法
    """

    def __init__(self, solver_config: PoissonConfig = None):
        """
        Args:
            solver_config: ポアソンソルバーの設定
        """
        # ポアソンソルバーの初期化
        self._poisson_solver = PoissonSolver(solver_config)

        # 診断情報
        self._diagnostics = {}

    def compute_rhs(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        dt: float,
    ) -> np.ndarray:
        """
        圧力ポアソン方程式の右辺を計算

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            dt: 時間刻み幅

        Returns:
            右辺項の配列
        """
        # 密度と速度の発散を計算
        div_rho_u = self._compute_density_velocity_divergence(velocity, density, dt)

        # 粘性の不均一性を考慮した粘性項を計算
        viscosity_term = self._compute_viscosity_term(velocity, viscosity)

        # 右辺を計算: -div(ρu)/dt + 粘性項
        rhs = -div_rho_u / dt + viscosity_term

        return rhs

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

    def solve_pressure(self, rhs: np.ndarray, velocity: VectorField) -> ScalarField:
        """
        圧力ポアソン方程式を解く

        Args:
            rhs: ポアソン方程式の右辺
            velocity: 速度場

        Returns:
            計算された圧力場
        """
        # ポアソンソルバーを使用して圧力を計算
        pressure = ScalarField(velocity.shape, velocity.dx)
        pressure.data = self._poisson_solver.solve(rhs)

        # 診断情報の更新
        self._diagnostics.update(self._poisson_solver.get_status())

        return pressure

    def project_velocity(
        self, velocity: VectorField, pressure: ScalarField
    ) -> VectorField:
        """
        圧力勾配を用いて速度場を修正（発散除去）

        Args:
            velocity: 元の速度場
            pressure: 計算された圧力場

        Returns:
            発散除去された速度場
        """
        dx = velocity.dx

        # 各方向に圧力勾配を減算
        for i in range(velocity.ndim):
            grad_p = np.gradient(pressure.data, dx, axis=i)
            velocity.components[i].data -= grad_p

        return velocity

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return self._diagnostics
