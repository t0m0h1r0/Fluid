from typing import Union, Optional
import jax.numpy as jnp
from jax import jit
from .base import Field, GridInfo
from .vector import VectorField


class ScalarField(Field):
    """3次元スカラー場"""

    def __init__(self, grid: GridInfo, initial_value: Union[float, jnp.ndarray] = 0.0):
        """
        スカラー場を初期化

        Args:
            grid: 計算グリッドの情報
            initial_value: 初期値（スカラーまたは配列）
        """
        super().__init__(grid)

        if jnp.isscalar(initial_value):
            self._data = jnp.full(grid.shape, float(initial_value))
        elif isinstance(initial_value, jnp.ndarray):
            if initial_value.shape != grid.shape:
                raise ValueError("初期値の形状がグリッドと一致しません")
            self._data = initial_value
        else:
            raise TypeError(f"未対応の初期値型: {type(initial_value)}")

    @property
    def data(self) -> jnp.ndarray:
        """フィールドデータを取得"""
        return self._data

    @jit
    def gradient(self, axis: Optional[int] = None) -> Union["ScalarField", VectorField]:
        """勾配を計算

        Args:
            axis: 方向（Noneの場合は全方向）

        Returns:
            勾配場（1方向の場合はScalarField、全方向の場合はVectorField）
        """
        if axis is not None:
            # 1方向の勾配
            grad_data = jnp.gradient(self._data, self.grid.dx[axis], axis=axis)
            return ScalarField(self.grid, grad_data)
        else:
            # 全方向の勾配
            grads = [
                jnp.gradient(self._data, d, axis=i) for i, d in enumerate(self.grid.dx)
            ]
            return VectorField(self.grid, grads)

    @jit
    def laplacian(self) -> "ScalarField":
        """ラプラシアンを計算"""
        result = jnp.zeros_like(self._data)
        for i, d in enumerate(self.grid.dx):
            result += jnp.gradient(jnp.gradient(self._data, d, axis=i), d, axis=i)
        return ScalarField(self.grid, result)

    def integrate(self) -> float:
        """領域全体の積分値を計算"""
        return float(jnp.sum(self._data * jnp.prod(self.grid.dx)))

    def copy(self) -> "ScalarField":
        """深いコピーを作成"""
        return ScalarField(self.grid, jnp.array(self._data))

    def __add__(self, other: Union["ScalarField", float]) -> "ScalarField":
        """加算演算子"""
        if jnp.isscalar(other):
            return ScalarField(self.grid, self._data + other)
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return ScalarField(self.grid, self._data + other._data)
        return NotImplemented

    def __sub__(self, other: Union["ScalarField", float]) -> "ScalarField":
        """減算演算子"""
        if jnp.isscalar(other):
            return ScalarField(self.grid, self._data - other)
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return ScalarField(self.grid, self._data - other._data)
        return NotImplemented

    def __mul__(
        self, other: Union["ScalarField", float, VectorField]
    ) -> Union["ScalarField", VectorField]:
        """乗算演算子

        スカラー場 * スカラー値
        スカラー場 * スカラー場
        スカラー場 * ベクトル場
        """
        if jnp.isscalar(other):
            # スカラー倍
            return ScalarField(self.grid, self._data * other)
        elif isinstance(other, ScalarField):
            # スカラー場同士の積
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return ScalarField(self.grid, self._data * other._data)
        elif isinstance(other, VectorField):
            # スカラー場とベクトル場の積
            return other * self
        return NotImplemented

    def __truediv__(self, other: Union["ScalarField", float]) -> "ScalarField":
        """除算演算子"""
        if jnp.isscalar(other):
            return ScalarField(self.grid, self._data / other)
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return ScalarField(self.grid, self._data / other._data)
        return NotImplemented

    def __pow__(self, power: Union[int, float]) -> "ScalarField":
        """累乗演算子"""
        return ScalarField(self.grid, self._data**power)

    def __neg__(self) -> "ScalarField":
        """単項マイナス演算子"""
        return ScalarField(self.grid, -self._data)

    def __eq__(self, other: "ScalarField") -> bool:
        """等価性比較"""
        if not isinstance(other, ScalarField):
            return NotImplemented
        if self.grid != other.grid:
            return False
        return jnp.array_equal(self._data, other._data)

    def __ne__(self, other: "ScalarField") -> bool:
        """非等価性比較"""
        result = self.__eq__(other)
        return not result if result is not NotImplemented else NotImplemented

    @jit
    def min(self) -> float:
        """最小値を取得"""
        return float(jnp.min(self._data))

    @jit
    def max(self) -> float:
        """最大値を取得"""
        return float(jnp.max(self._data))

    @jit
    def mean(self) -> float:
        """平均値を取得"""
        return float(jnp.mean(self._data))

    @jit
    def sum(self) -> float:
        """合計値を取得"""
        return float(jnp.sum(self._data))
