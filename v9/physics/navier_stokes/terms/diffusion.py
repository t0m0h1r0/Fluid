"""粘性項を実装するモジュール

このモジュールは、Navier-Stokes方程式の粘性項 ∇・(ν∇u) を実装します。
保存形式と非保存形式の両方の離散化を提供します。
"""

from typing import List, Dict, Any
import numpy as np

from core.field import VectorField
from physics.levelset import LevelSetField, LevelSetPropertiesManager
from .base import ViscousTerm


class DiffusionTerm(ViscousTerm):
    """粘性項クラス

    粘性による運動量の拡散を計算します。保存形式と非保存形式の
    両方の離散化をサポートします。
    """

    def __init__(
        self,
        use_conservative: bool = True,
        name: str = "Diffusion",
        enabled: bool = True,
        logger=None,
    ):
        """粘性項を初期化

        Args:
            use_conservative: 保存形式で離散化するかどうか
            name: 項の名前
            enabled: 項を有効にするかどうか
            logger: ロガー
        """
        super().__init__(name=name, enabled=enabled, logger=logger)
        self.use_conservative = use_conservative

    def compute(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: LevelSetPropertiesManager,
        **kwargs,
    ) -> List[np.ndarray]:
        """粘性項の寄与を計算

        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            **kwargs: 追加のパラメータ

        Returns:
            各方向の速度成分への寄与のリスト
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 粘性係数と密度の取得
        viscosity = properties.get_viscosity(levelset).data
        density = properties.get_density(levelset).data

        dx = velocity.dx
        result = []

        if self.use_conservative:
            # 保存形式での離散化
            # ∇・(μ∇u)の形式で計算
            for i, v_i in enumerate(velocity.components):
                # 変形速度テンソルの対角成分
                laplacian = np.zeros_like(v_i.data)

                # ∇・(μ∇u)の計算
                for j in range(velocity.ndim):
                    # ∂/∂xj(μ ∂ui/∂xj)
                    dui_dxj = np.gradient(v_i.data, dx, axis=j)
                    flux = viscosity * dui_dxj
                    laplacian += np.gradient(flux, dx, axis=j)

                # 密度で割って加速度に変換
                result.append(laplacian / density)

        else:
            # 非保存形式での離散化
            # ν∇²uの形式で計算
            kinematic_viscosity = viscosity / density

            for i, v_i in enumerate(velocity.components):
                # 速度のラプラシアン
                laplacian = sum(
                    np.gradient(np.gradient(v_i.data, dx, axis=j), dx, axis=j)
                    for j in range(velocity.ndim)
                )

                # 粘性による加速度
                result.append(kinematic_viscosity * laplacian)

        # 診断情報の更新
        total_dissipation = sum(
            np.sum(viscosity * self._compute_strain_rate_squared(velocity))
        )
        max_diffusion = max(np.max(np.abs(r)) for r in result)

        self._update_diagnostics("dissipation_rate", float(total_dissipation))
        self._update_diagnostics("max_diffusion", float(max_diffusion))
        self._update_diagnostics(
            "formulation",
            "conservative" if self.use_conservative else "non-conservative",
        )

        return result

    def _compute_strain_rate_squared(self, velocity: VectorField) -> np.ndarray:
        """変形速度テンソルの二乗ノルムを計算

        Args:
            velocity: 速度場

        Returns:
            変形速度テンソルの二乗ノルム |D|²
        """
        dx = velocity.dx
        result = np.zeros_like(velocity.components[0].data)

        # 対角成分の寄与
        for i, v_i in enumerate(velocity.components):
            dui_dxi = np.gradient(v_i.data, dx, axis=i)
            result += dui_dxi**2

        # 非対角成分の寄与
        for i in range(velocity.ndim):
            for j in range(i + 1, velocity.ndim):
                dui_dxj = np.gradient(velocity.components[i].data, dx, axis=j)
                duj_dxi = np.gradient(velocity.components[j].data, dx, axis=i)
                result += 0.5 * ((dui_dxj + duj_dxi) ** 2)

        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """項の診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "formulation": "conservative"
                if self.use_conservative
                else "non-conservative",
                "dissipation": {
                    "rate": self._diagnostics.get("dissipation_rate", 0.0),
                    "max": self._diagnostics.get("max_diffusion", 0.0),
                },
            }
        )
        return diag
