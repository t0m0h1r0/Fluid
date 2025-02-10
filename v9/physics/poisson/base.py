"""Poisson方程式ソルバーの基底クラスとプロトコル

このモジュールは、Poisson方程式を解くためのインターフェースと基底クラスを定義します。
"""

from abc import ABC, abstractmethod
from typing import Protocol, Optional, List, Dict, Any, Union
import numpy as np

from core.boundary import BoundaryCondition


class PoissonSolverConfig(Protocol):
    """Poissonソルバーの設定プロトコル"""

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        ...

    def get_config_for_component(self, component: str) -> Dict[str, Any]:
        """特定のコンポーネントの設定を取得"""
        ...


class PoissonSolverTerm(Protocol):
    """Poisson方程式の項のプロトコル"""

    @property
    def name(self) -> str:
        """項の名前"""
        ...

    def compute(self, solution: np.ndarray, **kwargs) -> np.ndarray:
        """項の寄与を計算"""
        ...

    def get_diagnostics(self, solution: np.ndarray, **kwargs) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


class PoissonSolverBase(ABC):
    """Poisson方程式ソルバーの基底抽象クラス"""

    def __init__(
        self,
        config: Optional[PoissonSolverConfig] = None,
        boundary_conditions: Optional[List[BoundaryCondition]] = None,
        logger=None,
    ):
        """ソルバーを初期化

        Args:
            config: ソルバー設定
            boundary_conditions: 境界条件
            logger: ロガー
        """
        self.config = config
        self.boundary_conditions = boundary_conditions or []
        self.logger = logger

        # 計算状態の追跡
        self._iteration_count = 0
        self._residual_history: List[float] = []
        self._converged = False

    @abstractmethod
    def solve(
        self, rhs: np.ndarray, initial_solution: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """Poisson方程式を解く"""
        pass

    @abstractmethod
    def compute_residual(
        self, solution: np.ndarray, rhs: np.ndarray, dx: Union[float, np.ndarray]
    ) -> float:
        """残差を計算"""
        pass

    def initialize(self, **kwargs):
        """ソルバーを初期化

        デフォルトの実装では基本的な状態をリセット
        サブクラスでオーバーライド可能
        """
        # 計算状態のリセット
        self._iteration_count = 0
        self._residual_history = []
        self._converged = False

        # ロギング
        if self.logger:
            self.logger.info(f"{self.__class__.__name__}ソルバーを初期化")

    def get_diagnostics(self) -> Dict[str, Any]:
        """ソルバーの診断情報を取得

        Returns:
            診断情報の辞書
        """
        return {
            "iteration_count": self._iteration_count,
            "converged": self._converged,
            "residual_history": self._residual_history,
            "final_residual": (
                self._residual_history[-1] if self._residual_history else None
            ),
        }

    def reset(self):
        """ソルバーの状態をリセット"""
        self._iteration_count = 0
        self._residual_history = []
        self._converged = False

    def log_diagnostics(self):
        """診断情報をログ出力"""
        if self.logger:
            diag = self.get_diagnostics()
            self.logger.info(f"Poissonソルバー診断情報: {diag}")
