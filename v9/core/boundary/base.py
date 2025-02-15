"""境界条件の基底クラスを提供するモジュール

このモジュールは、3次元流体計算における境界条件の基本的な抽象化を提供します。
すべての具体的な境界条件（Dirichlet、Neumann、周期境界など）は、
この基底クラスを継承します。
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Protocol
import jax.numpy as jnp
from jax import jit


class Side(Enum):
    """境界の側を表す列挙型"""

    NEGATIVE = auto()  # 負の側 (x=0, y=0, z=0)
    POSITIVE = auto()  # 正の側 (x=Lx, y=Ly, z=Lz)


class Direction(Enum):
    """座標軸の方向を表す列挙型"""

    X = 0
    Y = 1
    Z = 2


@dataclass(frozen=True)
class StencilInfo:
    """差分ステンシルの情報を保持する不変クラス

    Attributes:
        points: ステンシル点の相対位置（中心からのオフセット）
        coefficients: 各点での係数
    """

    points: jnp.ndarray  # shape: (N,)
    coefficients: jnp.ndarray  # shape: (N,)

    def __post_init__(self):
        """初期化後の検証"""
        if self.points.shape != self.coefficients.shape:
            raise ValueError("点の数と係数の数が一致しません")


class BoundaryField(Protocol):
    """境界条件が適用される場のプロトコル"""

    @property
    def shape(self) -> Tuple[int, int, int]:
        """3次元形状を取得"""
        ...

    def __array__(self) -> jnp.DeviceArray:
        """JAX配列としての表現を取得"""
        ...


class BoundaryCondition(ABC):
    """境界条件の基底抽象クラス"""

    def __init__(self, direction: Direction, side: Side, order: int = 2):
        """境界条件を初期化

        Args:
            direction: 境界面の方向（X, Y, Z）
            side: 境界の側（NEGATIVE, POSITIVE）
            order: 差分近似の次数（デフォルト: 2）
        """
        self.direction = direction
        self.side = side
        self.order = order

    @abstractmethod
    def apply(self, field: BoundaryField) -> jnp.DeviceArray:
        """境界条件を適用

        Args:
            field: 境界条件を適用する場

        Returns:
            境界条件が適用された新しい配列
        """
        pass

    @abstractmethod
    def get_stencil(self) -> StencilInfo:
        """差分ステンシルの情報を取得

        Returns:
            ステンシルの情報
        """
        pass

    @jit
    def get_boundary_slice(
        self, shape: Tuple[int, int, int], width: int = 1
    ) -> Tuple[slice, ...]:
        """境界領域のスライスを取得

        Args:
            shape: 場の3次元形状
            width: 境界領域の幅

        Returns:
            境界領域を選択するスライスのタプル
        """
        slices = [slice(None)] * 3
        if self.side == Side.NEGATIVE:
            slices[self.direction.value] = slice(0, width)
        else:
            slices[self.direction.value] = slice(-width, None)
        return tuple(slices)

    @jit
    def validate_field(self, field: BoundaryField) -> None:
        """場の妥当性を検証

        Args:
            field: 検証する場

        Raises:
            ValueError: 無効な場が指定された場合
        """
        if len(field.shape) != 3:
            raise ValueError("場は3次元である必要があります")
        if any(s <= 0 for s in field.shape):
            raise ValueError("場の各次元は正の大きさを持つ必要があります")
