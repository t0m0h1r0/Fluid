"""Navier-Stokes方程式の圧力項を提供するモジュール

このモジュールは、非圧縮性Navier-Stokes方程式の圧力項を実装します。
圧力勾配項を適切に計算し、密度の不連続性も考慮します。
"""

from typing import List, Dict, Any
import numpy as np

from core.field import VectorField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager
from .base import NavierStokesTerm


class PressureTerm(NavierStokesTerm):
    """圧力項クラス"""

    def __init__(self):
        """圧力項を初期化"""
        self._name = "Pressure"

    @property
    def name(self) -> str:
        """項の名前を取得"""
        return self._name

    def compute(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> List[np.ndarray]:
        """圧力項の寄与を計算

        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            kwargs:
                pressure: 圧力場（必須）

        Returns:
            各方向の速度成分への寄与のリスト
        """
        pressure = kwargs.get("pressure")
        if pressure is None:
            raise ValueError("Pressureが指定されていません")

        # 密度場の取得
        density = properties.get_density(levelset)

        # 結果を格納するリスト
        result = []

        # 各方向の圧力勾配を計算
        for i in range(velocity.ndim):
            # 圧力勾配の計算
            grad_p = np.gradient(pressure.data, velocity.dx, axis=i)

            # 密度で割って加速度に変換
            result.append(-grad_p / density.data)

        return result

    def get_diagnostics(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> Dict[str, Any]:
        """圧力項の診断情報を取得

        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            kwargs:
                pressure: 圧力場（オプション）

        Returns:
            診断情報を含む辞書
        """
        pressure = kwargs.get("pressure")
        if pressure is None:
            return {"type": "pressure", "pressure_available": False}

        # 圧力勾配の大きさを計算
        grad_p = np.array(
            [
                np.gradient(pressure.data, velocity.dx, axis=i)
                for i in range(velocity.ndim)
            ]
        )
        grad_p_mag = np.sqrt(np.sum(grad_p**2, axis=0))

        # 有効な値のみを使用
        valid_indices = np.isfinite(grad_p_mag)
        max_grad_p = np.max(grad_p_mag[valid_indices]) if np.any(valid_indices) else 0.0

        return {
            "type": "pressure",
            "pressure_available": True,
            "max_pressure": float(np.max(pressure.data)),
            "min_pressure": float(np.min(pressure.data)),
            "max_pressure_gradient": float(max_grad_p),
            "pressure_l2norm": float(
                np.sqrt(np.sum(pressure.data**2) * velocity.dx**velocity.ndim)
            ),
        }
