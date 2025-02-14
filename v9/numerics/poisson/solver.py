"""Poisson方程式のソルバー（改良版）"""

from typing import Optional, Union, List
import numpy as np

from core.field import ScalarField
from core.solver import IterativeSolver
from core.boundary import BoundaryCondition
from .base import PoissonSolverBase, PoissonSolverTerm
from .config import PoissonSolverConfig


class PoissonSolver(PoissonSolverBase, IterativeSolver):
    """Poisson方程式のソルバークラス（改良版）"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        boundary_conditions: Optional[List[BoundaryCondition]] = None,
        terms: Optional[List[PoissonSolverTerm]] = None,
        **kwargs,
    ):
        config = config or PoissonSolverConfig()
        kwargs.setdefault("name", "Poisson")
        kwargs.setdefault("tolerance", config.convergence.get("tolerance", 1e-6))
        kwargs.setdefault(
            "max_iterations", config.convergence.get("max_iterations", 1000)
        )

        PoissonSolverBase.__init__(
            self,
            config=config,
            boundary_conditions=boundary_conditions,
            logger=kwargs.get("logger"),
        )
        IterativeSolver.__init__(self, **kwargs)

        self.terms = terms or []
        self._converged = False
        self._initial_residual = None

    def compute_residual(
        self,
        solution: Union[np.ndarray, ScalarField],
        rhs: Union[np.ndarray, ScalarField],
        dx: Union[float, np.ndarray],
    ) -> float:
        """残差を計算（新しい演算子を活用）"""
        if isinstance(solution, ScalarField):
            solution_field = solution
        else:
            solution_field = ScalarField(solution.shape, dx, initial_value=solution)

        if isinstance(rhs, ScalarField):
            rhs_field = rhs
        else:
            rhs_field = ScalarField(rhs.shape, dx, initial_value=rhs)

        # ラプラシアンの計算（新しい演算子を活用）
        laplacian = solution_field.laplacian()

        # 残差の計算（新しい演算子を活用）
        residual = laplacian - rhs_field

        # 境界条件の適用
        if self.boundary_conditions:
            for i, bc in enumerate(self.boundary_conditions):
                if bc is not None:
                    residual = bc.apply_all(residual, i)

        # L2ノルムを計算（新しいnorm()メソッドを活用）
        return max(residual.norm(), 1e-15)

    def iterate(
        self,
        solution: Union[np.ndarray, ScalarField],
        rhs: Union[np.ndarray, ScalarField],
        dx: Union[float, np.ndarray],
    ) -> Union[np.ndarray, ScalarField]:
        """1回の反復を実行（新しい演算子を活用）"""
        if isinstance(solution, np.ndarray):
            solution = ScalarField(solution.shape, dx, initial_value=solution)
        if isinstance(rhs, np.ndarray):
            rhs = ScalarField(rhs.shape, dx, initial_value=rhs)

        # ラプラシアンの計算（新しい演算子を活用）
        laplacian = solution.laplacian()

        # 更新量の計算（新しい演算子を活用）
        update = (rhs - laplacian) * self.omega

        # 解の更新（新しい + 演算子を活用）
        return solution + update
