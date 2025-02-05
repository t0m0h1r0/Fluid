import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

from core.scheme import DifferenceScheme, BoundaryCondition


class AbstractPoissonSolver(ABC):
    """
    ポアソンソルバーの抽象基底クラス

    すべてのポアソンソルバーは以下の共通インターフェースを実装する必要があります。
    """

    def __init__(
        self,
        scheme: DifferenceScheme,
        boundary_conditions: List[BoundaryCondition],
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
    ):
        """
        コンストラクタ

        Args:
            scheme (DifferenceScheme): 差分スキーム
            boundary_conditions (List[BoundaryCondition]): 境界条件のリスト
            tolerance (float): 収束判定の許容誤差
            max_iterations (int): 最大反復回数
        """
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    @abstractmethod
    def solve(
        self, rhs: np.ndarray, initial_guess: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        ポアソン方程式を解くメソッド

        Args:
            rhs (np.ndarray): 右辺ベクトル
            initial_guess (Optional[np.ndarray]): 初期解（オプション）

        Returns:
            np.ndarray: 解
        """
        pass
