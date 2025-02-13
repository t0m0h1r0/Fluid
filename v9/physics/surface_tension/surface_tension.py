"""
表面張力の計算を担当するモジュール

このモジュールは、Level Set関数とそのデルタ関数を用いて、
二相流体における表面張力を計算するためのクラスを提供します。

表面張力の計算の理論:
F_st = σκδ(φ)n

ここで:
- σ: 表面張力係数
- κ: 平均曲率 (= -∇・(∇φ/|∇φ|))
- δ(φ): デルタ関数
- n: 界面の法線ベクトル (= ∇φ/|∇φ|)
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from physics.levelset import LevelSetField
from core.field import VectorField, ScalarField


class SurfaceTensionCalculator:
    """
    表面張力の計算を担当するクラス

    Level Set関数とそのデルタ関数を用いて、表面張力の空間分布と
    各方向への寄与を計算します。
    """

    def __init__(
        self,
        surface_tension_coefficient: float = 0.072,
        epsilon: Optional[float] = None,
    ):
        """
        表面張力計算器を初期化

        Args:
            surface_tension_coefficient: 表面張力係数 [N/m]
            epsilon: デルタ関数の正則化パラメータ（Noneの場合は自動設定）
        """
        self._sigma = surface_tension_coefficient
        self._epsilon = epsilon

    def compute_force(
        self, levelset: LevelSetField, **kwargs
    ) -> Tuple[VectorField, Dict[str, Any]]:
        """
        表面張力を計算

        Args:
            levelset: Level Set関数場
            **kwargs: 追加のパラメータ

        Returns:
            (表面張力のベクトル場, 診断情報)のタプル
        """
        # デルタ関数のパラメータ設定
        epsilon = self._epsilon or levelset.params.epsilon

        # 界面の法線ベクトルを計算
        normal = levelset.get_normal()  # これは既に正規化されている

        # 平均曲率の計算
        kappa = self._compute_mean_curvature(levelset)

        # デルタ関数の計算
        delta_field = levelset.get_delta()

        # 表面張力の計算: F_st = σκδ(φ)n
        force = VectorField(levelset.shape, levelset.dx)
        for i in range(levelset.ndim):
            # デルタ関数, 曲率, 表面張力係数を用いた力の計算
            force.components[i] = (
                -self._sigma
                * delta_field
                * ScalarField(levelset.shape, levelset.dx, kappa)
                * normal.components[i]
            )

        # 診断情報の作成
        diagnostics = {
            "surface_tension_coefficient": self._sigma,
            "max_force_magnitude": float(
                max(np.abs(comp.data).max() for comp in force.components)
            ),
            "max_curvature": float(np.max(np.abs(kappa))),
            "max_delta": float(np.max(delta_field.data)),
            "epsilon": epsilon,
        }

        return force, diagnostics

    def _compute_mean_curvature(self, levelset: LevelSetField) -> np.ndarray:
        """
        界面の平均曲率を計算: κ = -∇・(∇φ/|∇φ|)

        Args:
            levelset: Level Set関数場

        Returns:
            平均曲率の配列
        """
        # 既存の曲率計算メソッドを使用
        curvature_field = levelset.get_curvature(method="high_order")
        return curvature_field.data
