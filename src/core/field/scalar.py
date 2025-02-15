from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union
import jax
import jax.numpy as jnp

# JAXの64ビット浮動小数点数を有効化
jax.config.update("jax_enable_x64", True)

from .base import Field, FieldFactory, GridInfo

if TYPE_CHECKING:
    from .vector import VectorField


class ScalarField(Field):
    """3次元スカラー場"""

    def __init__(self, grid: GridInfo, initial_value: Union[float, jnp.ndarray] = 0.0):
        """スカラー場を初期化

        Args:
            grid: 計算グリッドの情報
            initial_value: 初期値（スカラーまたは配列）
        """
        super().__init__(grid)

        if jnp.isscalar(initial_value):
            self._data = jnp.full(grid.shape, float(initial_value), dtype=jnp.float64)
        elif hasattr(initial_value, "__array__"):  # numpy.ndarray などの配列型
            # JAX numpy 配列に変換
            try:
                data = jnp.asarray(initial_value, dtype=jnp.float64)
                if data.shape != grid.shape:
                    raise ValueError("初期値の形状がグリッドと一致しません")
                self._data = data
            except Exception as e:
                raise ValueError(f"配列の変換に失敗: {e}")
        else:
            raise TypeError(f"未対応の初期値型: {type(initial_value)}")

    @property
    def data(self) -> jnp.ndarray:
        """フィールドデータを取得"""
        return self._data

    def gradient(self, axis: Optional[int] = None) -> Union[ScalarField, 'VectorField']:
        """勾配を計算

        Args:
            axis: 方向（Noneの場合は全方向）

        Returns:
            勾配場（1方向の場合はScalarField、全方向の場合はVectorField）
        """
        if axis is not None:
            # 1方向の勾配（中心差分法、高次精度）
            dx = self.dx[axis]
            forward = jnp.roll(self._data, -1, axis=axis)
            backward = jnp.roll(self._data, 1, axis=axis)
            grad_data = (forward - backward) / (2 * dx)
            return FieldFactory.create_scalar_field(self.grid, grad_data)
        else:
            # 全方向の勾配
            grads = [
                jnp.gradient(self._data, d, axis=i) for i, d in enumerate(self.dx)
            ]
            return FieldFactory.create_vector_field(self.grid, tuple(grads))

    def divergence(self) -> ScalarField:
        """スカラー場に対する発散（ラプラシアン）を計算"""
        result = jnp.zeros_like(self._data, dtype=jnp.float64)
        for i, d in enumerate(self.dx):
            # 2階微分を中心差分で計算（高次精度）
            forward = jnp.roll(self._data, -1, axis=i)
            backward = jnp.roll(self._data, 1, axis=i)
            result += (forward - 2 * self._data + backward) / (d**2)
        return FieldFactory.create_scalar_field(self.grid, result)

    def integrate(self) -> float:
        """領域全体の積分値を計算"""
        # グリッド体積要素の計算
        dx_array = jnp.array(self.dx, dtype=jnp.float64)
        dv = jnp.prod(dx_array)
        return float(jnp.sum(self._data) * dv)

    def copy(self) -> ScalarField:
        """深いコピーを作成"""
        return FieldFactory.create_scalar_field(self.grid, jnp.array(self._data))

    def __add__(self, other: Union[ScalarField, float]) -> ScalarField:
        """加算演算子"""
        if jnp.isscalar(other):
            return FieldFactory.create_scalar_field(
                self.grid, self._data + jnp.float64(other)
            )
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return FieldFactory.create_scalar_field(self.grid, self._data + other._data)
        return NotImplemented

    def __sub__(self, other: Union[ScalarField, float]) -> ScalarField:
        """減算演算子"""
        if jnp.isscalar(other):
            return FieldFactory.create_scalar_field(
                self.grid, self._data - jnp.float64(other)
            )
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            return FieldFactory.create_scalar_field(self.grid, self._data - other._data)
        return NotImplemented

    def __mul__(
        self, other: Union[ScalarField, float, 'VectorField']
    ) -> Union[ScalarField, 'VectorField']:
        """乗算演算子"""
        if jnp.isscalar(other):
            return FieldFactory.create_scalar_field(
                self.grid, self._data * jnp.float64(other)
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
        if jnp.isscalar(other):
            return FieldFactory.create_scalar_field(
                self.grid, self._data / jnp.float64(other)
            )
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            # ゼロ除算を防ぐための微小値を追加
            epsilon = jnp.finfo(jnp.float64).eps
            return FieldFactory.create_scalar_field(
                self.grid, self._data / (other._data + epsilon)
            )
        return NotImplemented

    def __pow__(self, power: Union[int, float]) -> ScalarField:
        """累乗演算子"""
        return FieldFactory.create_scalar_field(
            self.grid, jnp.power(self._data, jnp.float64(power))
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
        return jnp.array_equal(self._data, other._data)

    def min(self) -> float:
        """最小値を取得"""
        return float(jnp.min(self._data))

    def max(self) -> float:
        """最大値を取得"""
        return float(jnp.max(self._data))

    def mean(self) -> float:
        """平均値を取得"""
        return float(jnp.mean(self._data))

    def sum(self) -> float:
        """合計値を取得"""
        return float(jnp.sum(self._data))

    def norm(self, p: int = 2) -> float:
        """p-ノルムを計算"""
        return float(jnp.linalg.norm(self._data.ravel(), ord=p))