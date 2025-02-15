from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as np

if TYPE_CHECKING:
    from .scalar import ScalarField
    from .vector import VectorField

# 型変数の定義
FieldType = TypeVar("FieldType", bound="Field")
ComponentType = TypeVar("ComponentType", np.ndarray, list[np.ndarray])


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


class Field(ABC):
    """Field の基底抽象クラス"""

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

    @property
    def shape(self) -> Tuple[int, int, int]:
        """形状を取得"""
        return self.grid.shape

    @property
    def dx(self) -> Tuple[float, float, float]:
        """グリッド間隔を取得"""
        return self.grid.dx

    @property
    @abstractmethod
    def data(self) -> ComponentType:
        """フィールドデータを取得"""
        pass

    @abstractmethod
    def gradient(self, axis: int | None = None) -> Field:
        """勾配を計算"""
        pass

    @abstractmethod
    def divergence(self) -> ScalarField:
        """発散を計算"""
        pass

    @abstractmethod
    def copy(self) -> Field:
        """深いコピーを作成"""
        pass

    def norm(self, p: int = 2) -> float:
        """p-ノルムを計算

        Args:
            p: ノルムの次数（デフォルト: 2）

        Returns:
            計算されたノルム値
        """
        data_array = np.asarray(self.data)
        return float(np.linalg.norm(data_array.ravel(), ord=p))

    def __repr__(self) -> str:
        """文字列表現"""
        return f"{self.__class__.__name__}(shape={self.shape}, time={self.grid.time})"


class FieldFactory:
    """フィールドの生成を担当するファクトリークラス"""

    @staticmethod
    def create_scalar_field(
        grid: GridInfo, initial_value: float | np.ndarray = 0.0
    ) -> ScalarField:
        """ScalarField インスタンスを生成"""
        from .scalar import ScalarField

        return ScalarField(grid, initial_value)

    @staticmethod
    def create_vector_field(
        grid: GridInfo, initial_values: Tuple[float | np.ndarray, ...] = (0.0, 0.0, 0.0)
    ) -> VectorField:
        """VectorField インスタンスを生成"""
        from .vector import VectorField

        return VectorField(grid, initial_values)
