import numpy as np
from .field import LevelSetField, LevelSetParameters
from typing import Optional


class LevelSetInitializer:
    """Level Set関数の初期設定を担当"""

    @staticmethod
    def create_background_layer(
        shape: tuple,
        height_fraction: float,
        dx: float = 1.0,
        params: Optional[LevelSetParameters] = None,
    ) -> LevelSetField:
        """背景水層を初期化"""
        z = np.linspace(0, 1, shape[2])
        phi = height_fraction - z
        return LevelSetField(
            data=np.tile(phi, (shape[0], shape[1], 1)), dx=dx, params=params
        )

    @staticmethod
    def create_sphere(
        shape: tuple,
        center: np.ndarray,
        radius: float,
        dx: float = 1.0,
        params: Optional[LevelSetParameters] = None,
    ) -> LevelSetField:
        """球オブジェクトを初期化"""
        x = np.linspace(0, 1, shape[0])
        y = np.linspace(0, 1, shape[1])
        z = np.linspace(0, 1, shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        phi = (
            np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
            - radius
        )

        return LevelSetField(data=phi, dx=dx, params=params)
