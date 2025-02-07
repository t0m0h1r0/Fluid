"""周期境界条件を提供するモジュール

このモジュールは、流体シミュレーションで使用される周期境界条件を実装します。
周期境界条件では、計算領域の両端が接続されているとして扱います。
"""

import numpy as np
from .base import BoundaryCondition, StencilInfo


class PeriodicBoundary(BoundaryCondition):
    """周期境界条件クラス

    周期境界条件は、計算領域の両端が接続されているとして扱います。
    物理量は領域の境界で連続的に繰り返されます。
    """

    def apply(self, field: np.ndarray, axis: int, side: int) -> np.ndarray:
        """周期境界条件を適用

        Args:
            field: 境界条件を適用する場
            axis: 境界条件を適用する軸
            side: 境界の側（0: 負側、1: 正側）

        Returns:
            境界条件が適用された場
        """
        self.validate_field(field, axis)
        result = field.copy()

        # ステンシル情報から必要なゴースト点の数を決定
        stencil = self.get_stencil(side)
        n_ghost = len(stencil.points)

        # 境界領域のスライスを取得
        ghost_slice = self.get_boundary_slice(field, axis, side, n_ghost)

        # 反対側の内部領域から値をコピー
        if side == 0:  # 負側の境界
            source_slice = self.get_boundary_slice(field, axis, 1, n_ghost)
            result[ghost_slice] = field[source_slice]
        else:  # 正側の境界
            source_slice = self.get_boundary_slice(field, axis, 0, n_ghost)
            result[ghost_slice] = field[source_slice]

        return result

    def get_stencil(self, side: int) -> StencilInfo:
        """差分ステンシルの情報を取得

        Args:
            side: 境界の側（0: 負側、1: 正側）

        Returns:
            ステンシルの情報
        """
        # 2次精度中心差分の場合
        if self.order == 2:
            return StencilInfo(
                points=np.array([-1, 0, 1]), coefficients=np.array([-0.5, 0.0, 0.5])
            )
        # 4次精度中心差分の場合
        elif self.order == 4:
            return StencilInfo(
                points=np.array([-2, -1, 0, 1, 2]),
                coefficients=np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
            )
        else:
            raise ValueError(f"未対応の次数です: {self.order}")
