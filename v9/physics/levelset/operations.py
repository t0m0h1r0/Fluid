import numpy as np
from .field import LevelSetField


class LevelSetOperations:
    """Level Set関数に対する基本的な操作を提供"""

    @staticmethod
    def advect(
        levelset: LevelSetField, velocity: np.ndarray, dt: float
    ) -> LevelSetField:
        """Level Set関数を移流"""
        advected_data = levelset.data.copy()

        for i in range(levelset.shape[0]):
            for j in range(levelset.shape[1]):
                for k in range(levelset.shape[2]):
                    # Lagrange的な移流計算（簡略化）
                    x, y, z = i * levelset.dx, j * levelset.dx, k * levelset.dx
                    u, v, w = (
                        velocity[0][i, j, k],
                        velocity[1][i, j, k],
                        velocity[2][i, j, k],
                    )

                    x_prev = x - u * dt
                    y_prev = y - v * dt
                    z_prev = z - w * dt

                    # バイリニア補間
                    advected_data[i, j, k] = np.interp(
                        x_prev,
                        np.linspace(
                            0, levelset.shape[0] * levelset.dx, levelset.shape[0]
                        ),
                        levelset.data[i, j, :],
                    )

        return LevelSetField(data=advected_data, dx=levelset.dx, params=levelset.params)

    @staticmethod
    def dilate(levelset: LevelSetField, distance: float) -> LevelSetField:
        """Level Set関数を膨張"""
        dilated_data = levelset.data - distance
        return LevelSetField(data=dilated_data, dx=levelset.dx, params=levelset.params)

    @staticmethod
    def erode(levelset: LevelSetField, distance: float) -> LevelSetField:
        """Level Set関数を収縮"""
        eroded_data = levelset.data + distance
        return LevelSetField(data=eroded_data, dx=levelset.dx, params=levelset.params)
