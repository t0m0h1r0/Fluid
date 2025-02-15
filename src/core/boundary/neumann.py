"""Neumann境界条件を提供するモジュール

このモジュールは、3次元流体計算におけるNeumann境界条件を実装します。
境界上で物理量の勾配を指定する境界条件を提供します。
"""

from typing import Union, Callable
import jax.numpy as jnp
from jax import jit

from .base import (
    BoundaryCondition,
    BoundaryField,
    Direction,
    Side,
    StencilInfo,
)


class NeumannBC(BoundaryCondition):
    """Neumann境界条件クラス

    境界上で法線方向の勾配を指定する境界条件:
    ∂f/∂n|_boundary = g(x, y, z)

    ここで g は指定された勾配値または勾配を計算する関数です。
    """

    def __init__(
        self,
        direction: Direction,
        side: Side,
        gradient: Union[float, Callable[[float, float, float], float]],
        order: int = 2,
    ):
        """Neumann境界条件を初期化

        Args:
            direction: 境界面の方向（X, Y, Z）
            side: 境界の側（NEGATIVE, POSITIVE）
            gradient: 勾配値または勾配を計算する関数
            order: 差分近似の次数（デフォルト: 2）
        """
        super().__init__(direction, side, order)
        self._gradient = gradient

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

        # 境界とその近傍のスライスを取得
        boundary_slice = self.get_boundary_slice(field.shape)
        inner_slice = self._get_inner_slice(field.shape)

        # 境界上での勾配値の計算
        if callable(self._gradient):
            coords = self._get_boundary_coordinates(field.shape)
            gradient_values = self._gradient(*coords)
        else:
            gradient_values = self._gradient

        # 境界値の更新（2次精度の中心差分を使用）
        dx = 1.0  # 正規化された格子間隔
        if self.side == Side.NEGATIVE:
            # f_0 = f_1 - grad * dx
            result = result.at[boundary_slice].set(
                result[inner_slice] - gradient_values * dx
            )
        else:
            # f_N = f_{N-1} + grad * dx
            result = result.at[boundary_slice].set(
                result[inner_slice] + gradient_values * dx
            )

        return result

    def get_stencil(self) -> StencilInfo:
        """差分ステンシルの情報を取得

        Returns:
            ステンシルの情報（2次または4次精度）
        """
        if self.order == 2:
            # 2次精度
            if self.side == Side.NEGATIVE:
                points = jnp.array([0, 1])
                coeffs = jnp.array([-1.0, 1.0])
            else:
                points = jnp.array([-1, 0])
                coeffs = jnp.array([-1.0, 1.0])
        elif self.order == 4:
            # 4次精度
            if self.side == Side.NEGATIVE:
                points = jnp.array([0, 1, 2, 3])
                coeffs = jnp.array([-11 / 6, 3, -3 / 2, 1 / 3])
            else:
                points = jnp.array([-3, -2, -1, 0])
                coeffs = jnp.array([-1 / 3, 3 / 2, -3, 11 / 6])
        else:
            raise ValueError(f"未対応の次数です: {self.order}")

        return StencilInfo(points, coeffs)

    @jit
    def _get_inner_slice(self, shape: tuple[int, int, int]) -> tuple[slice, ...]:
        """境界の内側のスライスを取得

        Args:
            shape: 場の3次元形状

        Returns:
            内側領域を選択するスライスのタプル
        """
        slices = [slice(None)] * 3
        if self.side == Side.NEGATIVE:
            slices[self.direction.value] = slice(1, 2)
        else:
            slices[self.direction.value] = slice(-2, -1)
        return tuple(slices)

    @jit
    def _get_boundary_coordinates(
        self, shape: tuple[int, int, int]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """境界面の座標グリッドを生成

        Args:
            shape: 場の3次元形状

        Returns:
            (x, y, z) 座標のタプル
        """
        # グリッドの生成
        nx, ny, nz = shape
        x = jnp.linspace(0, 1, nx)
        y = jnp.linspace(0, 1, ny)
        z = jnp.linspace(0, 1, nz)

        # 境界面のスライス位置
        pos = 0 if self.side == Side.NEGATIVE else -1

        # 方向に応じた座標グリッドの生成
        if self.direction == Direction.X:
            coord = x[pos]
            yy, zz = jnp.meshgrid(y, z, indexing="ij")
            xx = jnp.full_like(yy, coord)
        elif self.direction == Direction.Y:
            coord = y[pos]
            xx, zz = jnp.meshgrid(x, z, indexing="ij")
            yy = jnp.full_like(xx, coord)
        else:  # Direction.Z
            coord = z[pos]
            xx, yy = jnp.meshgrid(x, y, indexing="ij")
            zz = jnp.full_like(xx, coord)

        return xx, yy, zz
