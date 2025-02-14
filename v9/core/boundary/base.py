"""3次元の物理場を表現する基底クラスを提供するモジュール

このモジュールは、JAXを使用した3次元の物理場の基本的な抽象化を提供します。
すべての具体的な場の実装（スカラー場、ベクトル場）は、この基底クラスを継承します。
"""

from abc import ABC, abstractmethod
from typing import Tuple, Protocol, Dict, Any
from dataclasses import dataclass
import jax.numpy as jnp
from jax import jit


# 場の値を保持するプロトコル
class FieldValue(Protocol):
    """場の値の型プロトコル"""

    @property
    def shape(self) -> Tuple[int, int, int]:
        """3次元形状を取得"""
        ...

    def __array__(self) -> jnp.DeviceArray:
        """JAX配列としての表現を取得"""
        ...


@dataclass(frozen=True)
class GridInfo:
    """計算グリッドの情報を保持する不変クラス

    Attributes:
        shape: グリッドの3次元形状 (nx, ny, nz)
        dx: 各方向のグリッド間隔 (dx, dy, dz)
        time: 現在の時刻
    """

    shape: Tuple[int, int, int]
    dx: Tuple[float, float, float]
    time: float = 0.0

    def __post_init__(self):
        """初期化後の検証"""
        if len(self.shape) != 3 or len(self.dx) != 3:
            raise ValueError("GridInfoは3次元データのみ対応しています")
        if any(s <= 0 for s in self.shape):
            raise ValueError("グリッドサイズは正の値である必要があります")
        if any(d <= 0 for d in self.dx):
            raise ValueError("グリッド間隔は正の値である必要があります")
        if self.time < 0:
            raise ValueError("時刻は非負である必要があります")


class Field(ABC):
    """3次元物理場の基底抽象クラス"""

    def __init__(self, grid: GridInfo):
        """場を初期化

        Args:
            grid: 計算グリッドの情報
        """
        self._grid = grid

    @property
    def grid(self) -> GridInfo:
        """グリッド情報を取得"""
        return self._grid

    @property
    @abstractmethod
    def data(self) -> FieldValue:
        """場のデータを取得する抽象プロパティ"""
        pass

    @abstractmethod
    def copy(self) -> "Field":
        """場の深いコピーを作成する抽象メソッド"""
        pass

    @jit
    def norm(self, p: int = 2) -> float:
        """場のp-ノルムを計算

        Args:
            p: ノルムの次数（デフォルト: 2）

        Returns:
            計算されたノルム値
        """
        return float(jnp.linalg.norm(jnp.asarray(self.data).ravel(), ord=p))

    def get_diagnostics(self) -> Dict[str, Any]:
        """場の診断情報を取得

        Returns:
            診断情報を含む辞書
        """
        data_array = jnp.asarray(self.data)
        return {
            "shape": self.grid.shape,
            "dx": self.grid.dx,
            "time": self.grid.time,
            "min": float(data_array.min()),
            "max": float(data_array.max()),
            "mean": float(data_array.mean()),
            "norm": float(self.norm()),
        }
