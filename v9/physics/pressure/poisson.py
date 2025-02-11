"""二相流体における圧力ポアソン方程式（PPE）のソルバー

圧力ポアソン方程式の導出:
1. Navier-Stokes方程式の両辺の発散を取る
2. 非圧縮性条件 ∇・u = 0 を適用
3. 密度 ρ の空間変化を考慮し、調和平均で補間
4. 外力（重力、界面張力）の影響を考慮

結果として得られる方程式:
∇・(1/ρ ∇p) = ∇・(u・∇u) + ∇・fs - ∇・(∇ν・∇u)

ここで:
- p: 圧力
- ρ: 密度（空間的に変化）
- u: 速度場
- ν: 動粘性係数
- fs: 界面張力による体積力
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from core.field import VectorField, ScalarField
from numerics.poisson import PoissonConfig
#from numerics.poisson import PoissonSolver
from numerics.poisson import SORSolver as PoissonSolver

class PressurePoissonSolver:
    """圧力ポアソン方程式のソルバー"""

    def __init__(self, solver_config: Optional[PoissonConfig] = None):
        """
        Args:
            solver_config: ポアソンソルバーの設定
        """
        self._poisson_solver = PoissonSolver(solver_config or PoissonConfig())
        self._diagnostics = {}

    def solve(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        dt: float,
        external_force: Optional[VectorField] = None,
        initial_pressure: Optional[ScalarField] = None,
    ) -> Tuple[ScalarField, Dict[str, Any]]:
        """圧力場を計算

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            dt: 時間刻み幅
            external_force: 外力場（オプション）
            initial_pressure: 初期圧力場（オプション）

        Returns:
            (圧力場, 診断情報)のタプル
        """
        # 右辺の計算
        rhs = self._compute_rhs(velocity, density, viscosity, dt, external_force)

        # 圧力場の計算
        pressure = ScalarField(velocity.shape, velocity.dx)
        if initial_pressure is not None:
            pressure.data = initial_pressure.data.copy()

        pressure.data = self._poisson_solver.solve(rhs, initial_solution=pressure.data)

        return pressure, self._diagnostics

    def _compute_rhs(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        dt: float,
        external_force: Optional[VectorField] = None,
    ) -> np.ndarray:
        """右辺項を計算

        各項の寄与:
        1. 移流項: ∇・(u・∇u)
        2. 粘性項: -∇・(∇ν・∇u)
        3. 外力項: ∇・f
        """
        dx = velocity.dx
        rhs = np.zeros_like(density.data)

        # 移流項の計算
        for i, v_i in enumerate(velocity.components):
            for j, v_j in enumerate(velocity.components):
                grad_vi = np.gradient(v_i.data, dx, axis=j)
                rhs += np.gradient(v_j.data * grad_vi, dx, axis=j)

        # 粘性の不均一性を考慮した項の計算
        viscosity_term = self._compute_viscosity_term(velocity, viscosity)
        rhs -= viscosity_term

        # 外力項の計算
        if external_force is not None:
            for i, f_i in enumerate(external_force.components):
                rhs += np.gradient(f_i.data, dx, axis=i)

        # 密度の逆数による調和平均
        rhs *= self._compute_inverse_density(density)

        return rhs

    def _compute_viscosity_term(
        self, velocity: VectorField, viscosity: ScalarField
    ) -> np.ndarray:
        """粘性の不均一性を考慮した項を計算"""
        dx = velocity.dx
        result = np.zeros_like(viscosity.data)

        # 粘性勾配の計算
        nu_grad = []
        for i in range(velocity.ndim):
            nu_grad.append(np.gradient(viscosity.data, dx, axis=i))

        # 速度勾配との積を計算
        for i in range(velocity.ndim):
            vel_grad = []
            for j in range(velocity.ndim):
                vel_grad.append(np.gradient(velocity.components[i].data, dx, axis=j))

            for j in range(velocity.ndim):
                result += np.gradient(nu_grad[j] * vel_grad[j], dx, axis=i)

        return result

    def _compute_inverse_density(self, density: ScalarField) -> np.ndarray:
        """密度の逆数を調和平均で計算"""
        # セル中心での密度の逆数を計算
        rho = np.maximum(density.data, 1e-10)  # ゼロ除算防止
        inv_rho = 1.0 / rho

        # 調和平均の計算（必要に応じて）
        # 現在は単純な逆数を返す
        return inv_rho

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return self._diagnostics
