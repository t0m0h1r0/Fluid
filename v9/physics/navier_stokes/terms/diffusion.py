"""粘性項を実装するモジュール

このモジュールは、Navier-Stokes方程式の粘性項 ∇・(ν∇u) を実装します。
"""

from typing import List, Dict, Any
import numpy as np
from core.field import VectorField
from ..base import NavierStokesTerm


class DiffusionTerm(NavierStokesTerm):
    """粘性項クラス"""

    def __init__(self, use_conservative: bool = True):
        """粘性項を初期化

        Args:
            use_conservative: 保存形式で離散化するかどうか
        """
        self._name = "Diffusion"
        self.use_conservative = use_conservative

    @property
    def name(self) -> str:
        """項の名前を取得"""
        return self._name

    def compute(self, velocity: VectorField, dt: float, **kwargs) -> List[np.ndarray]:
        """粘性項の寄与を計算

        Args:
            velocity: 現在の速度場
            dt: 時間刻み幅
            **kwargs: 追加のパラメータ
                - viscosity: 粘性係数場
                - density: 密度場

        Returns:
            各方向の速度成分への寄与のリスト
        """
        # 物性値の取得
        properties = kwargs.get("properties", None)
        levelset = kwargs.get("levelset", None)

        # 粘性係数と密度の設定
        if properties is not None and levelset is not None:
            viscosity = properties.get_viscosity(levelset)
            density = properties.get_density(levelset)
            mu = viscosity.data
            rho = density.data
        else:
            mu = kwargs.get("viscosity", np.ones_like(velocity.components[0].data))
            rho = kwargs.get("density", np.ones_like(velocity.components[0].data))

        dx = velocity.dx
        result = []

        if self.use_conservative:
            # 保存形式での離散化
            for i, v_i in enumerate(velocity.components):
                # 変形速度テンソルの対角成分
                laplacian = np.zeros_like(v_i.data)

                # ∇・(μ∇u)の計算
                for j in range(velocity.ndim):
                    # ∂/∂xj(μ ∂ui/∂xj)
                    dui_dxj = np.gradient(v_i.data, dx, axis=j)
                    flux = mu * dui_dxj
                    laplacian += np.gradient(flux, dx, axis=j)

                # 密度で割って加速度に変換
                result.append(laplacian / rho)

        else:
            # 非保存形式での離散化
            for i, v_i in enumerate(velocity.components):
                # 速度のラプラシアン
                laplacian = sum(
                    np.gradient(np.gradient(v_i.data, dx, axis=j), dx, axis=j)
                    for j in range(velocity.ndim)
                )

                # 粘性による加速度
                result.append(mu * laplacian / rho)

        return result

    def get_diagnostics(self, velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """粘性項の診断情報を取得"""
        # 粘性による散逸の計算
        dissipation = self.compute(velocity, 1.0, **kwargs)
        max_dissipation = max(np.max(np.abs(d)) for d in dissipation)
        mean_dissipation = np.mean([np.mean(np.abs(d)) for d in dissipation])

        return {
            "formulation": "conservative"
            if self.use_conservative
            else "non-conservative",
            "max_dissipation": float(max_dissipation),
            "mean_dissipation": float(mean_dissipation),
        }
