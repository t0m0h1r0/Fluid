"""Dirichlet境界条件を提供するモジュール

このモジュールは、3次元流体計算におけるDirichlet境界条件を実装します。
境界上で物理量の値を直接指定する境界条件を提供します。
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


class DirichletBC(BoundaryCondition):
    """Dirichlet境界条件クラス

    境界上で値を直接指定する境界条件:
    f|_boundary = g(x, y, z)

    ここで g は指定された境界値または境界値を計算する関数です。
    """

    def __init__(
        self,
        direction: Direction,
        side: Side,
        value: Union[float, Callable[[float, float, float], float]],
        order: int = 2,
    ):
        """Dirichlet境界条件を初期化

        Args:
            direction: 境界面の方向（X, Y, Z）
            side: 境界の側（NEGATIVE, POSITIVE）
            value: 境界値または境界値を計算する関数
            order: 差分近似の次数（デフォルト: 2）
        """
        super().__init__(direction, side, order)
        self._value = value

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

        # 境界スライスの取得
        boundary_slice = self.get_boundary_slice(field.shape)

        # 境界値の計算
        if callable(self._value):
            # 境界面の座標グリッドを生成
            coords = self._get_boundary_coordinates(field.shape)
            boundary_values = self._value(*coords)
        else:
            boundary_values = self._value

        # 境界値の設定
        result = result.at[boundary_slice].set(boundary_values)

        return result

    def get_stencil(self) -> StencilInfo:
        """差分ステンシルの情報を取得

        Returns:
            ステンシルの情報（2次または4次精度）
        """
        if self.order == 2:
            # 2次精度
            if self.side == Side.NEGATIVE:
                points = jnp.array([0, 1, 2])
                coeffs = jnp.array([-3 / 2, 2, -1 / 2])
            else:
                points = jnp.array([-2, -1, 0])
                coeffs = jnp.array([1 / 2, -2, 3 / 2])
        elif self.order == 4:
            # 4次精度
            if self.side == Side.NEGATIVE:
                points = jnp.array([0, 1, 2, 3, 4])
                coeffs = jnp.array([-25 / 12, 4, -3, 4 / 3, -1 / 4])
            else:
                points = jnp.array([-4, -3, -2, -1, 0])
                coeffs = jnp.array([1 / 4, -4 / 3, 3, -4, 25 / 12])
        else:
            raise ValueError(f"未対応の次数です: {self.order}")

        return StencilInfo(points, coeffs)

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
