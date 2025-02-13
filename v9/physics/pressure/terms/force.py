"""
圧力ポアソン方程式の外力項を計算するモジュール

外力項 ∇⋅f の計算を実装します。
"""

import numpy as np
from typing import Optional

from core.field import VectorField, ScalarField
from .base import PoissonTerm


class ForceTerm(PoissonTerm):
    """外力項の計算クラス"""

    def __init__(self, name: str = "ExternalForce", enabled: bool = True):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)

    def compute(
        self,
        shape: tuple,
        dx: float,
        external_force: Optional[VectorField] = None,
        **kwargs,
    ) -> ScalarField:
        """
        外力項 ∇⋅f を計算

        Args:
            shape: 計算領域の形状
            dx: グリッド間隔
            external_force: 外力場（オプション）

        Returns:
            外力項の発散を表すScalarField
        """
        if not self.enabled or external_force is None:
            return ScalarField(shape, dx)

        # 外力場の発散を計算
        result = external_force.divergence()

        # 診断情報の更新
        self._update_diagnostics(result, external_force)

        return result

    def _update_diagnostics(self, result: ScalarField, force: VectorField):
        """
        診断情報を更新

        Args:
            result: 計算された外力項
            force: 外力場
        """
        self._diagnostics = {
            "divergence": {
                "max": float(np.max(np.abs(result.data))),
                "min": float(np.min(result.data)),
                "mean": float(np.mean(result.data)),
                "norm": float(np.linalg.norm(result.data)),
            },
            "force_components": {
                f"component_{i}": {
                    "max": float(np.max(np.abs(comp.data))),
                    "min": float(np.min(comp.data)),
                }
                for i, comp in enumerate(force.components)
            },
        }
