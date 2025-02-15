from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import jax.numpy as np

from .base import Field, FieldFactory, GridInfo

if TYPE_CHECKING:
    from .vector import VectorField


class ScalarField(Field):
    """3次元スカラー場"""

    def __init__(self, grid: GridInfo, initial_value: Union[float, np.ndarray] = 0.0):
        """スカラー場を初期化

        Args:
            grid: 計算グリッドの情報
            initial_value: 初期値（スカラーまたは配列）
        """
        super().__init__(grid)

        if np.isscalar(initial_value):
            self._data = np.full(grid.shape, float(initial_value))
        elif isinstance(initial_value, (np.ndarray, list)):
            # リストの場合はnumpy配列に変換
            data = np.array(initial_value, dtype=np.float64)
            if data.shape != grid.shape:
                raise ValueError("初期値の形状がグリッドと一致しません")
            self._data = data
        else:
            raise TypeError(f"未対応の初期値型: {type(initial_value)}")

    @property
    def data(self) -> np.ndarray:
        """フィールドデータを取得"""
        return self._data

    def gradient(self, axis: Optional[int] = None) -> Union[ScalarField, VectorField]:
        """勾配を計算

        Args:
            axis: 方向（Noneの場合は全方向）

        Returns:
            勾配場（1方向の場合はScalarField、全方向の場合はVectorField）
        """
        if axis is not None:
            # 1方向の勾配（中心差分法、高次精度）
            dx = self.dx[axis]
            grad_data = np.gradient(self._data, dx, axis=axis, edge_order=2)
            return FieldFactory.create_scalar_field(self.grid, grad_data)
        else:
            # 全方向の勾配
            grads = [
                np.gradient(self._data, d, axis=i, edge_order=2)
                for i, d in enumerate(self.dx)
            ]
            return FieldFactory.create_vector_field(self.grid, tuple(grads))

    def divergence(self) -> ScalarField:
        """スカラー場に対する発散（ラプラシアン）を計算"""
        result = np.zeros_like(self._data)
        for i, d in enumerate(self.dx):
            # 2階微分を中心差分で計算（高次精度）
            result += np.gradient(
                np.gradient(self._data, d, axis=i, edge_order=2),
                d,
                axis=i,
                edge_order=2,
            )
        return FieldFactory.create_scalar_field(self.grid, result)

    def integrate(self) -> float:
        """領域全体の積分値を計算"""
        # グリッド体積要素の計算
        dx_array = np.array(self.dx, dtype=np.float64)
        dv = np.prod(dx_array)
        return float(np.sum(self._data) * dv)

    def copy(self) -> ScalarField:
        """深いコピーを作成"""
        return FieldFactory.create_scalar_field(self.grid, np.array(self._data))

    def __add__(self, other: Union[ScalarField, float]) -> ScalarField:
        """加算演算子"""
        if np.isscalar(other):
            return FieldFactory.create_scalar_field(
                self.grid, self._data + np.float64(other)
            )
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return FieldFactory.create_scalar_field(self.grid, self._data + other._data)
        return NotImplemented

    def __sub__(self, other: Union[ScalarField, float]) -> ScalarField:
        """減算演算子"""
        if np.isscalar(other):
            return FieldFactory.create_scalar_field(
                self.grid, self._data - np.float64(other)
            )
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return FieldFactory.create_scalar_field(self.grid, self._data - other._data)
        return NotImplemented

    def __mul__(
        self, other: Union[ScalarField, float, VectorField]
    ) -> Union[ScalarField, VectorField]:
        """乗算演算子"""
        if np.isscalar(other):
            return FieldFactory.create_scalar_field(
                self.grid, self._data * np.float64(other)
            )
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return FieldFactory.create_scalar_field(self.grid, self._data * other._data)
        elif isinstance(other, VectorField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            components = [comp.data * self._data for comp in other.components]
            return FieldFactory.create_vector_field(self.grid, tuple(components))
        return NotImplemented

    def __truediv__(self, other: Union[ScalarField, float]) -> ScalarField:
        """除算演算子"""
        if np.isscalar(other):
            return FieldFactory.create_scalar_field(
                self.grid, self._data / np.float64(other)
            )
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            # ゼロ除算を防ぐための微小値を追加
            epsilon = np.finfo(np.float64).eps
            return FieldFactory.create_scalar_field(
                self.grid, self._data / (other._data + epsilon)
            )
        return NotImplemented

    def __pow__(self, power: Union[int, float]) -> ScalarField:
        """累乗演算子"""
        return FieldFactory.create_scalar_field(
            self.grid, np.power(self._data, np.float64(power))
        )

    def __neg__(self) -> ScalarField:
        """単項マイナス演算子"""
        return FieldFactory.create_scalar_field(self.grid, -self._data)

    def __eq__(self, other: ScalarField) -> bool:
        """等価性比較"""
        if not isinstance(other, ScalarField):
            return NotImplemented
        if self.grid != other.grid:
            return False
        return np.array_equal(self._data, other._data)

    def min(self) -> float:
        """最小値を取得"""
        return float(np.min(self._data))

    def max(self) -> float:
        """最大値を取得"""
        return float(np.max(self._data))

    def mean(self) -> float:
        """平均値を取得"""
        return float(np.mean(self._data))

    def sum(self) -> float:
        """合計値を取得"""
        return float(np.sum(self._data))

    def norm(self, p: int = 2) -> float:
        """p-ノルムを計算"""
        return float(np.linalg.norm(self._data.ravel(), ord=p))
