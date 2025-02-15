"""周期境界条件を提供するモジュール

このモジュールは、3次元流体計算における周期境界条件を実装します。
計算領域の両端を接続する境界条件を提供します。
"""

import jax.numpy as jnp
from jax import jit

from .base import (
    BoundaryCondition,
    BoundaryField,
    Direction,
    Side,
    StencilInfo,
)


class PeriodicBC(BoundaryCondition):
    """周期境界条件クラス

    計算領域の両端を接続する境界条件:
    f(x + L) = f(x)

    ここで L は計算領域の大きさです。
    """

    def __init__(self, direction: Direction, side: Side, order: int = 2):
        """周期境界条件を初期化

        Args:
            direction: 境界面の方向（X, Y, Z）
            side: 境界の側（NEGATIVE, POSITIVE）
            order: 差分近似の次数（デフォルト: 2）
        """
        super().__init__(direction, side, order)

    @jit
    def apply(self, field: BoundaryField) -> jnp.DeviceArray:
        """境界条件を適用

        Args:
            field: 境界条件を適用する場

        Returns:
            境界条件が適用された新しい配列
        """
        self.validate_field(field)
        result = jnp.array(field)

        # 境界のスライス幅を取得
        stencil = self.get_stencil()
        width = len(stencil.points)

        # 境界と対応する内部領域のスライスを取得
        boundary_slice = self.get_boundary_slice(field.shape, width)
        opposite_slice = self._get_opposite_slice(field.shape, width)

        # 値のコピー
        if self.side == Side.NEGATIVE:
            # 負の側の境界には、領域の終端の値をコピー
            result = result.at[boundary_slice].set(result[opposite_slice])
        else:
            # 正の側の境界には、領域の始端の値をコピー
            result = result.at[boundary_slice].set(result[opposite_slice])

        return result

    def get_stencil(self) -> StencilInfo:
        """差分ステンシルの情報を取得

        Returns:
            ステンシルの情報（2次または4次精度）
        """
        if self.order == 2:
            # 2次精度中心差分
            points = jnp.array([-1, 0, 1])
            coeffs = jnp.array([-0.5, 0.0, 0.5])
        elif self.order == 4:
            # 4次精度中心差分
            points = jnp.array([-2, -1, 0, 1, 2])
            coeffs = jnp.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12])
        else:
            raise ValueError(f"未対応の次数です: {self.order}")

        return StencilInfo(points, coeffs)

    @jit
    def _get_opposite_slice(
        self, shape: tuple[int, int, int], width: int
    ) -> tuple[slice, ...]:
        """反対側の境界領域のスライスを取得

        Args:
            shape: 場の3次元形状
            width: 境界領域の幅

        Returns:
            反対側の境界領域を選択するスライスのタプル
        """
        slices = [slice(None)] * 3
        if self.side == Side.NEGATIVE:
            # 負の側の境界の場合、終端からwidth個の点を選択
            slices[self.direction.value] = slice(-2 * width, -width)
        else:
            # 正の側の境界の場合、始端からwidth個の点を選択
            slices[self.direction.value] = slice(width, 2 * width)
        return tuple(slices)
