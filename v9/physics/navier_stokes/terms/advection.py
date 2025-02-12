from typing import Dict, Any
import numpy as np

from core.field import VectorField
from .base import BaseNavierStokesTerm


class AdvectionTerm(BaseNavierStokesTerm):
    """移流項クラス

    中心差分法を用いて移流項を計算します。
    高次精度が必要な場合は、numerics/spatial/ から適切なスキームをインポートして使用してください。
    """

    def __init__(
        self,
        name: str = "Advection",
        enabled: bool = True,
    ):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)
        self._scheme = "central"  # デフォルトは中心差分

    def compute(self, velocity: VectorField, **kwargs) -> VectorField:
        """移流項の寄与を計算

        Args:
            velocity: 速度場

        Returns:
            各方向の速度成分への移流項の寄与をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 結果用のVectorFieldを作成
        result = VectorField(velocity.shape, velocity.dx)

        # 各方向の移流項を計算
        for i, v_i in enumerate(velocity.components):
            # 中心差分による移流項の計算
            flux = -sum(
                v_j.data * v_i.gradient(j) for j, v_j in enumerate(velocity.components)
            )
            # 結果をVectorFieldのコンポーネントに設定
            result.components[i].data = flux

        # 診断情報の更新
        self._diagnostics["flux_max"] = float(
            max(np.max(np.abs(comp.data)) for comp in result.components)
        )
        self._diagnostics["scheme"] = self._scheme

        return result

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """移流項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 最大速度の計算
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # CFL条件に基づく時間刻み幅の計算
        cfl = kwargs.get("cfl", 0.5)
        return cfl * velocity.dx / (max_velocity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "scheme": self._scheme,
                "flux_max": self._diagnostics.get("flux_max", 0.0),
            }
        )
        return diag
