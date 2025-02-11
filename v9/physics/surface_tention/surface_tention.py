"""
表面張力の計算を担当するモジュール

このモジュールは、Level Set関数とそのデルタ関数を用いて、
二相流体における表面張力を計算するためのクラスを提供します。
"""

from typing import List, Optional, Tuple, Dict, Any
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

    def compute_force(self, levelset: LevelSetField, **kwargs) -> List[np.ndarray]:
        """
        表面張力を計算

        Args:
            levelset: Level Set関数場
            **kwargs: 追加のパラメータ

        Returns:
            各方向の表面張力成分
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
        force_components = []
        for axis in range(levelset.ndim):
            # デルタ関数, 曲率, 表面張力係数を用いた力の計算
            force = -self._sigma * delta * kappa * normals[axis]
            force_components.append(force)

        return force_components

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

    def get_diagnostics(
        self, force_components: List[np.ndarray], levelset: LevelSetField
    ) -> Dict[str, Any]:
        """
        表面張力に関する診断情報を取得

        Args:
            force_components: 各方向の表面張力成分
            levelset: Level Set関数場

        Returns:
            診断情報の辞書
        """
        return {
            "surface_tension_coefficient": self._sigma,
            "max_force_magnitude": np.max([np.abs(f).max() for f in force_components]),
            "interface_area": levelset.area(),
            "interface_volume": levelset.volume(),
            "epsilon": self._epsilon or levelset.params.epsilon,
        }


def compute_surface_tension_force(
    levelset: LevelSetField, surface_tension_coefficient: float = 0.072
) -> Tuple[VectorField, Dict[str, Any]]:
    """
    表面張力を計算し、VectorFieldとして返す便利関数

    Args:
        levelset: Level Set関数場
        surface_tension_coefficient: 表面張力係数 [N/m]

    Returns:
        表面張力のベクトル場と診断情報のタプル
    """
    # 表面張力計算器の初期化
    surface_tension_calculator = SurfaceTensionCalculator(surface_tension_coefficient)

    # 表面張力の計算
    force_components = surface_tension_calculator.compute_force(levelset)

    # VectorFieldに変換
    surface_tension_force = VectorField(levelset.shape, levelset.dx)
    for i, comp in enumerate(surface_tension_force.components):
        comp.data = force_components[i]

    # 診断情報の取得
    diagnostics = surface_tension_calculator.get_diagnostics(force_components, levelset)

    return surface_tension_force, diagnostics
