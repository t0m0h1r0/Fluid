"""
粘性項（拡散項）の計算を提供するモジュール（改良版）

Navier-Stokes方程式における∇⋅(μ(∇u+∇uᵀ)) 項を計算します。
新しい演算子とメソッドを活用して実装を改善しています。
"""

from typing import Dict, Any, Union
import numpy as np

from core.field import VectorField, ScalarField
from .base import BaseNavierStokesTerm


class DiffusionTerm(BaseNavierStokesTerm):
    """粘性項（拡散項）を計算するクラス（改良版）"""

    def __init__(self, name: str = "Diffusion", enabled: bool = True, order: int = 2):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
            order: 差分近似の次数
        """
        super().__init__(name, enabled)
        self._order = order

    def compute(
        self,
        velocity: VectorField,
        viscosity: Union[float, ScalarField, None] = None,
        **kwargs,
    ) -> VectorField:
        """
        粘性項 ∇⋅(μ(∇u+∇uᵀ)) を計算
        新しい演算子を活用して実装を改善

        Args:
            velocity: 速度場
            viscosity: 粘性係数（スカラー値またはスカラー場）

        Returns:
            拡散項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 粘性係数の正規化（新しいコンストラクタを活用）
        nu_field = (
            viscosity
            if isinstance(viscosity, ScalarField)
            else ScalarField(
                velocity.shape, velocity.dx, initial_value=viscosity or 1.0e-3
            )
        )

        # 対称勾配テンソルの計算（新しいメソッドを活用）
        strain = velocity.symmetric_gradient()

        # 粘性応力の計算と発散（新しい演算子を活用）
        result = (nu_field * strain).divergence()

        # 診断情報の更新
        self._update_diagnostics(result, nu_field)
        return result

    def _update_diagnostics(self, result: VectorField, viscosity: ScalarField):
        """診断情報を更新（新しいメソッドを活用）"""
        self._diagnostics = {
            "order": self._order,
            "diffusion": {
                "max_magnitude": float(result.magnitude().max()),
                "component_max": {
                    f"component_{i}": float(comp.norm(ord=np.inf))
                    for i, comp in enumerate(result.components)
                },
            },
            "viscosity": {
                "min": float(viscosity.min()),
                "max": float(viscosity.max()),
                "mean": float(viscosity.mean()),
            },
        }

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """
        拡散項に基づく時間刻み幅の制限を計算
        新しいメソッドを活用

        Args:
            velocity: 速度場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 粘性の取得と正規化
        viscosity = kwargs.get("viscosity", 1.0e-3)
        if isinstance(viscosity, ScalarField):
            viscosity = viscosity.max()

        # von Neumannの安定性解析に基づく制限
        dx_min = min(velocity.dx)
        return 0.5 * dx_min * dx_min / (viscosity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag
