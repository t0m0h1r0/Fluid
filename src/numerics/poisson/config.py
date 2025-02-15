from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class PoissonSolverConfig:
    """Poisson方程式ソルバーの設定を管理するクラス"""

    # 収束判定パラメータ
    max_iterations: int = 1000
    tolerance: float = 1e-6
    absolute_tolerance: bool = False
    relaxation_parameter: float = 1.0

    # グリッド情報
    dx: np.ndarray | None = None

    # 診断情報の設定
    diagnostics: Dict[str, Any] = field(
        default_factory=lambda: {"save_residual_history": True, "log_frequency": 10}
    )

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.max_iterations <= 0:
            raise ValueError("最大反復回数は正の整数である必要があります")

        if self.tolerance <= 0:
            raise ValueError("収束判定の許容誤差は正の値である必要があります")

        if not 0 < self.relaxation_parameter <= 2:
            raise ValueError("緩和パラメータは0から2の間である必要があります")
