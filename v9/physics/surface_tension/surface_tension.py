"""
表面張力の計算を担当するモジュール

このモジュールは、Level Set関数とそのデルタ関数を用いて、
二相流体における表面張力を計算するためのクラスを提供します。
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from physics.levelset import LevelSetField
from core.field import VectorField


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

        # Level Set関数の勾配と勾配の大きさを計算
        grad_values = []
        grad_norm = np.zeros_like(levelset.data)

        for axis in range(levelset.ndim):
            grad_ax = np.gradient(levelset.data, levelset.dx, axis=axis)
            grad_values.append(grad_ax)
            grad_norm += grad_ax**2

        grad_norm = np.sqrt(grad_norm + 1e-10)

        # デルタ関数と法線ベクトルの計算
        delta = 0.5 / epsilon * (1.0 - np.tanh(levelset.data / epsilon) ** 2)
        normals = [g / grad_norm for g in grad_values]

        # 曲率の計算
        kappa = self._compute_curvature(levelset, grad_norm)

        # 表面張力の計算（法線方向に垂直）
        force = VectorField(levelset.shape, levelset.dx)
        for i in range(levelset.ndim):
            # デルタ関数, 曲率, 表面張力係数を用いた力の計算
            force.components[i].data = -self._sigma * delta * kappa * normals[i]

        # 診断情報の作成
        diagnostics = {
            "surface_tension_coefficient": self._sigma,
            "max_force_magnitude": float(
                max(np.abs(comp.data).max() for comp in force.components)
            ),
            "max_curvature": float(np.max(np.abs(kappa))),
            "epsilon": epsilon,
        }

        return force, diagnostics

    def _compute_curvature(
        self, levelset: LevelSetField, grad_norm: np.ndarray
    ) -> np.ndarray:
        """
        界面の曲率を計算

        Args:
            levelset: Level Set関数場
            grad_norm: 勾配の大きさ

        Returns:
            曲率の配列
        """
        # 勾配の発散（曲率）の計算
        kappa = np.zeros_like(levelset.data)
        for i in range(levelset.ndim):
            # 各次元の曲率成分を計算
            grad_norm_derivative = np.gradient(
                grad_norm / (grad_norm + 1e-10), levelset.dx, axis=i
            )
            kappa += grad_norm_derivative

        return kappa
