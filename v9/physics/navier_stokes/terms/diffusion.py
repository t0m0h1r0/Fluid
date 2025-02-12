from typing import Dict, Any
import numpy as np

from core.field import VectorField
from .base import BaseNavierStokesTerm


class DiffusionTerm(BaseNavierStokesTerm):
    """粘性項（拡散項）クラス"""

    def __init__(
        self, viscosity: float = 1.0e-3, name: str = "Diffusion", enabled: bool = True
    ):
        """
        Args:
            viscosity: 動粘性係数
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)
        self._viscosity = viscosity

    def compute(self, velocity: VectorField, **kwargs) -> VectorField:
        """粘性項の寄与を計算

        Args:
            velocity: 速度場

        Returns:
            各方向の速度成分への拡散項の寄与をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 結果用のVectorFieldを作成
        result = VectorField(velocity.shape, velocity.dx)
        dx = velocity.dx

        # 各成分の粘性項を計算
        for i, v_i in enumerate(velocity.components):
            # ラプラシアンの計算
            laplacian = sum(
                np.gradient(np.gradient(v_i.data, dx, axis=j), dx, axis=j)
                for j in range(velocity.ndim)
            )

            # 粘性による加速度を結果に設定
            result.components[i].data = self._viscosity * laplacian

        # 診断情報の更新
        total_dissipation = self._compute_dissipation(velocity)
        self._diagnostics = {
            "total_dissipation": float(total_dissipation),
            "viscosity": self._viscosity,
        }

        return result

    def _compute_dissipation(self, velocity: VectorField) -> float:
        """粘性散逸の計算

        Args:
            velocity: 速度場

        Returns:
            粘性散逸の総和
        """
        strain_rate_squared = np.zeros_like(velocity.components[0].data)
        dx = velocity.dx

        # 速度勾配の計算
        for i in range(velocity.ndim):
            for j in range(velocity.ndim):
                # ひずみ速度テンソルの計算
                if i == j:
                    # 対角成分
                    dui_dxi = np.gradient(velocity.components[i].data, dx, axis=i)
                    strain_rate_squared += dui_dxi**2
                else:
                    # 非対角成分
                    dui_dxj = np.gradient(velocity.components[i].data, dx, axis=j)
                    duj_dxi = np.gradient(velocity.components[j].data, dx, axis=i)
                    strain_rate_squared += 2 * ((dui_dxj + duj_dxi) / 2) ** 2

        return np.sum(self._viscosity * strain_rate_squared)

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """粘性項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 安定性条件: dx² / (2ν)
        return 0.5 * velocity.dx**2 / (self._viscosity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "viscosity": self._viscosity,
                "dissipation": self._diagnostics.get("total_dissipation", 0.0),
            }
        )
        return diag
