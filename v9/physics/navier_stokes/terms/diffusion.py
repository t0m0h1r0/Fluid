"""
粘性項（拡散項）の計算を提供するモジュール

Navier-Stokes方程式における∇⋅(μ(∇u+∇uT)) 項を計算します。
"""

from typing import Dict, Any, Union
import numpy as np

from core.field import VectorField, ScalarField
from .base import BaseNavierStokesTerm


class DiffusionTerm(BaseNavierStokesTerm):
    """
    粘性項（拡散項）を計算するクラス

    速度場の粘性拡散を中心差分で近似計算します。
    """

    def __init__(
        self, 
        name: str = "Diffusion", 
        enabled: bool = True,
        order: int = 2
    ):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
            order: 差分近似の次数（2次、4次など）
        """
        super().__init__(name, enabled)
        self._order = order
        self._diagnostics: Dict[str, Any] = {}

    def compute(
        self, 
        velocity: VectorField, 
        viscosity: Union[float, ScalarField, None] = None,
        **kwargs
    ) -> VectorField:
        """
        粘性項の寄与を計算

        Args:
            velocity: 速度場
            viscosity: 粘性係数（定数、スカラー場、またはNone）

        Returns:
            拡散項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        # 結果用のVectorFieldを作成
        result = VectorField(velocity.shape, velocity.dx)
        dx = velocity.dx

        # 粘性係数の設定
        nu = 1.0e-3 if viscosity is None else viscosity

        # 各方向の拡散項を計算
        dissipation_terms = []
        for i, v_i in enumerate(velocity.components):
            # 粘性係数の処理（スカラー場または定数）
            if isinstance(nu, ScalarField):
                effective_nu = nu.data
            else:
                effective_nu = nu

            # ラプラシアンの計算（対称勾配テンソル: ∇⋅(μ(∇u+∇uT))）
            laplacian = np.zeros_like(v_i.data)
            for j in range(velocity.ndim):
                # 対角成分: 2 * ∂μ/∂x_j * ∂u_i/∂x_j
                dui_dxj = np.gradient(v_i.data, dx, axis=j)
                
                # 粘性係数が空間的に変化する場合の追加項
                if isinstance(nu, ScalarField):
                    dnu_dxj = np.gradient(effective_nu, dx, axis=j)
                    laplacian += dnu_dxj * dui_dxj
                
                # 2次精度中心差分のラプラシアン
                laplacian += effective_nu * (
                    np.gradient(np.gradient(v_i.data, dx, axis=j), dx, axis=j)
                )

            # 結果をVectorFieldに設定
            result.components[i].data = laplacian
            dissipation_terms.append(laplacian)

        # 診断情報の更新
        self._update_diagnostics(result, dissipation_terms)

        return result

    def _update_diagnostics(
        self, 
        result: VectorField, 
        dissipation_terms: list
    ):
        """
        診断情報を更新

        Args:
            result: 計算された拡散項
            dissipation_terms: 各成分の拡散項
        """
        max_dissipation = [np.max(np.abs(term)) for term in dissipation_terms]
        self._diagnostics = {
            "order": self._order,
            "max_dissipation_x": float(max_dissipation[0]) if len(max_dissipation) > 0 else 0.0,
            "max_dissipation_y": float(max_dissipation[1]) if len(max_dissipation) > 1 else 0.0,
            "max_dissipation_z": float(max_dissipation[2]) if len(max_dissipation) > 2 else 0.0,
            "max_diffusion": float(
                max(np.max(np.abs(comp.data)) for comp in result.components)
            ),
        }

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """
        拡散項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場

        Returns:
            計算された時間刻み幅の制限
        """
        if not self.enabled:
            return float("inf")

        # 粘性の取得（デフォルト値）
        viscosity = kwargs.get('viscosity', 1.0e-3)
        
        # 拡散項に基づく時間刻み幅の制限: dt ≤ dx² / (2ν)
        return 0.5 * velocity.dx**2 / (viscosity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag