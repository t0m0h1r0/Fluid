"""
二相流体における圧力ポアソン方程式（PPE）のソルバー

圧力ポアソン方程式の導出:
1. 密度変化を考慮したラプラシアン形の圧力ポアソン方程式
2. 密度の勾配項を近似的に無視
∇²p ≈ ρ(∇・(u・∇u) + ∇・fs - ∇・(∇ν・∇u))
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from core.field import VectorField, ScalarField
from numerics.poisson import PoissonConfig

# from numerics.poisson import PoissonSolver
# from numerics.poisson import SORSolver as PoissonSolver
from numerics.poisson import ConjugateGradientSolver as PoissonSolver


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
        force: Optional[VectorField] = None,
        initial_solution: Optional[ScalarField] = None,
        **kwargs
    ) -> Tuple[ScalarField, Dict[str, Any]]:
        """
        圧力ポアソン方程式を解く

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            force: 外力場（オプション）
            initial_solution: 初期推定解（オプション）

        Returns:
            (圧力場, 診断情報)のタプル
        """
        # 密度と次元数を取得
        rho = density.data
        dx = velocity.dx
        ndim = velocity.ndim

        # 右辺の非密度項の計算
        rhs = np.zeros_like(rho)

        # 速度の発散項（密度スケールなし）
        for i in range(ndim):
            flux = np.zeros_like(rho)
            for j in range(ndim):
                # u_j * ∂u_i/∂x_j の計算
                flux += (
                    velocity.components[j].data * 
                    np.gradient(velocity.components[i].data, dx, axis=j)
                )
            rhs += flux

        # 外力項の追加（オプション、密度スケールなし）
        if force is not None:
            for i in range(ndim):
                rhs += np.gradient(force.components[i].data, force.dx, axis=i)

        # 最後に密度をかける
        rhs *= rho

        # 初期解のセットアップ
        if initial_solution is None:
            initial_solution = ScalarField(velocity.shape, velocity.dx)

        # ポアソンソルバーの実行
        result_data = self._poisson_solver.solve(rhs, initial_solution=initial_solution.data)

        # 圧力場の作成
        pressure = ScalarField(velocity.shape, velocity.dx)
        pressure.data = result_data

        # 診断情報の更新
        diagnostics = {}
        if hasattr(self._poisson_solver, "get_diagnostics"):
            diagnostics.update(self._poisson_solver.get_diagnostics())
        
        # メモリへの診断情報の保存
        self._diagnostics = diagnostics

        return pressure, diagnostics

    def compute_residual(
        self, solution: ScalarField, rhs: np.ndarray, **kwargs
    ) -> float:
        """残差を計算

        Args:
            solution: 現在の解
            rhs: 右辺

        Returns:
            計算された残差
        """
        # ラプラシアンの計算
        laplacian = np.zeros_like(solution.data)
        for axis in range(solution.ndim):
            forward = np.roll(solution.data, 1, axis=axis)
            backward = np.roll(solution.data, -1, axis=axis)
            laplacian += (forward + backward - 2 * solution.data) / (solution.dx ** 2)

        # 残差の計算
        residual = laplacian - rhs

        # L2ノルムを計算
        residual_norm = np.sqrt(np.mean(residual**2))
        return max(residual_norm, 1e-15)  # 最小値を保証

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return self._diagnostics