"""圧力項を実装するモジュール

このモジュールは、Navier-Stokes方程式の圧力項を実装します。
圧力勾配項を適切に計算し、密度の不連続性も考慮します。
"""

from typing import List, Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField, LevelSetPropertiesManager
from .base import TermBase


class PressureTerm(TermBase):
    """圧力項クラス"""

    def __init__(
        self,
        use_conservative: bool = True,
        name: str = "Pressure",
        enabled: bool = True,
        logger=None,
    ):
        """圧力項を初期化

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
        """圧力項の寄与を計算

        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            **kwargs: 追加のパラメータ
                - pressure: 圧力場（必須）

        Returns:
            各方向の速度成分への寄与のリスト
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 圧力場の取得
        pressure = kwargs.get("pressure")
        if pressure is None:
            raise ValueError("圧力場が指定されていません")
        if not isinstance(pressure, ScalarField):
            raise TypeError("pressureはScalarField型である必要があります")

        # 密度場の取得
        density = properties.get_density(levelset).data

        if self.use_conservative:
            # 保存形式での離散化
            # ∇・(p/ρ I)の形式で計算
            result = self._compute_conservative(pressure, density, velocity.dx)
        else:
            # 非保存形式での離散化
            # (1/ρ)∇pの形式で計算
            result = self._compute_non_conservative(pressure, density, velocity.dx)

        # 診断情報の更新
        self._update_diagnostics(pressure, density, result)

        return result

    def _compute_conservative(
        self, pressure: ScalarField, density: np.ndarray, dx: float
    ) -> List[np.ndarray]:
        """保存形式での圧力項の計算"""
        result = []

        for i in range(pressure.ndim):
            # 圧力/密度の勾配を計算
            p_over_rho = pressure.data / density
            grad_p_rho = np.gradient(p_over_rho, dx, axis=i)

            # 負の勾配を取る（運動方程式の右辺項として）
            result.append(-grad_p_rho)

        return result

    def _compute_non_conservative(
        self, pressure: ScalarField, density: np.ndarray, dx: float
    ) -> List[np.ndarray]:
        """非保存形式での圧力項の計算"""
        result = []

        for i in range(pressure.ndim):
            # 圧力勾配を計算
            grad_p = np.gradient(pressure.data, dx, axis=i)

            # 密度で割って加速度に変換
            result.append(-grad_p / density)

        return result

    def _update_diagnostics(
        self,
        pressure: ScalarField,
        density: np.ndarray,
        contributions: List[np.ndarray],
    ):
        """診断情報を更新"""
        # 圧力勾配の大きさを計算
        grad_p_mag = np.sqrt(sum(np.sum(c**2) for c in contributions))

        # 圧力仕事を計算
        pressure_work = np.sum(pressure.data * np.sum(contributions))

        # 診断情報を更新
        self._diagnostics.update(
            {
                "formulation": "conservative"
                if self.use_conservative
                else "non-conservative",
                "pressure": {
                    "min": float(np.min(pressure.data)),
                    "max": float(np.max(pressure.data)),
                    "mean": float(np.mean(pressure.data)),
                    "gradient_magnitude": float(grad_p_mag),
                },
                "work": float(pressure_work),
            }
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """項の診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "formulation": "conservative"
                if self.use_conservative
                else "non-conservative",
                "pressure": self._diagnostics.get("pressure", {}),
                "work": self._diagnostics.get("work", 0.0),
            }
        )
        return diag

    def compute_timestep(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: LevelSetPropertiesManager,
        **kwargs,
    ) -> float:
        """圧力項に基づく時間刻み幅の制限を計算

        音速に基づくCFL条件を実装。
        ただし、非圧縮性流れでは通常この制限は使用しない。
        """
        if not self.enabled:
            return float("inf")

        # 圧力場の取得
        pressure = kwargs.get("pressure")
        if pressure is None:
            return float("inf")

        # 音速の見積もり（非圧縮性近似のため、大きな値を使用）
        density = properties.get_density(levelset)
        c = np.sqrt(np.max(pressure.data) / np.min(density.data))

        # CFL条件に基づく時間刻み幅の制限
        dx = velocity.dx
        return 0.5 * dx / c
