from typing import Tuple, Protocol
import jax.numpy as jnp
from dataclasses import dataclass
from jax import jit


@dataclass(frozen=True)
class GridInfo:
    """3D計算グリッドの不変情報を表現"""

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


class FieldData(Protocol):
    """フィールドデータの型プロトコル"""

    @property
    def shape(self) -> Tuple[int, int, int]:
        """3次元形状を取得"""
        ...

    def __array__(self) -> jnp.ndarray:
        """JAX配列への変換"""
        ...


class Field:
    """3次元物理場の基底抽象クラス"""

    def __init__(self, grid: GridInfo):
        """
        Args:
            grid: 計算グリッドの情報
        """
        self._grid = grid

    @property
    def grid(self) -> GridInfo:
        """グリッド情報を取得"""
        return self._grid

    @jit
    def norm(self, p: int = 2) -> float:
        """p-ノルムを計算

        Args:
            p: ノルムの次数（デフォルト: 2）

        Returns:
            計算されたノルム値
        """
        data_array = jnp.asarray(self.data)
        return float(jnp.linalg.norm(data_array.ravel(), ord=p))

    def __repr__(self) -> str:
        """文字列表現"""
        return (
            f"{self.__class__.__name__}(shape={self.grid.shape}, time={self.grid.time})"
        )
