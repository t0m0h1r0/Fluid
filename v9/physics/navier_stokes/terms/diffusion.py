"""
粘性項（拡散項）の計算を提供するモジュール

Navier-Stokes方程式における∇⋅(μ(∇u+∇uᵀ)) 項を計算します。
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

    def __init__(self, name: str = "Diffusion", enabled: bool = True, order: int = 2):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
            order: 差分近似の次数（2次、4次など）
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
        各成分 [∇⋅(μ(∇u_i+∇u_i^T))]_i を計算

        Args:
            velocity: 速度場
            viscosity: 粘性係数（スカラー場または定数）

        Returns:
            拡散項をVectorFieldとして返す
        """
        if not self.enabled:
            return VectorField(velocity.shape, velocity.dx)

        result = VectorField(velocity.shape, velocity.dx)
        dx = velocity.dx

        # 粘性係数の設定（スカラー場または定数）
        nu = viscosity if viscosity is not None else 1.0e-3
        nu_field = (
            nu
            if isinstance(nu, ScalarField)
            else ScalarField(velocity.shape, dx, initial_value=nu)
        )

        for i, v_i in enumerate(velocity.components):
            diffusion = np.zeros_like(v_i.data)

            for j in range(velocity.ndim):
                # ∂u_i/∂x_j を計算
                dui_dxj = v_i.gradient(j)

                # ∂u_j/∂x_i を計算 (対称項)
                if i != j:  # 非対角項の場合のみ
                    duj_dxi = velocity.components[j].gradient(i)
                    strain_rate = 0.5 * (dui_dxj + duj_dxi)
                else:
                    strain_rate = dui_dxj

                # 粘性応力項の発散を計算
                # ∂/∂x_j(μ * 2ε_ij)
                stress = nu_field.data * (2.0 * strain_rate)
                diffusion += np.gradient(stress, dx, axis=j)

            result.components[i].data = diffusion

        # 診断情報の更新
        self._update_diagnostics(result, nu_field)

        return result

    def _update_diagnostics(self, result: VectorField, viscosity: ScalarField):
        """
        診断情報を更新

        Args:
            result: 計算された拡散項
            viscosity: 粘性係数場
        """
        diffusion_max = [float(np.max(np.abs(comp.data))) for comp in result.components]

        self._diagnostics = {
            "order": self._order,
            "max_diffusion_x": diffusion_max[0] if len(diffusion_max) > 0 else 0.0,
            "max_diffusion_y": diffusion_max[1] if len(diffusion_max) > 1 else 0.0,
            "max_diffusion_z": diffusion_max[2] if len(diffusion_max) > 2 else 0.0,
            "max_diffusion": float(max(diffusion_max)),
            "viscosity_range": {
                "min": float(np.min(viscosity.data)),
                "max": float(np.max(viscosity.data)),
                "mean": float(np.mean(viscosity.data)),
            },
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
        viscosity = kwargs.get("viscosity", 1.0e-3)
        if isinstance(viscosity, ScalarField):
            viscosity = np.max(viscosity.data)

        # 拡散項に基づく時間刻み幅の制限: dt ≤ dx² / (2ν)
        return 0.5 * velocity.dx**2 / (viscosity + 1e-10)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(self._diagnostics)
        return diag
