from typing import Union, List, Tuple
import jax.numpy as jnp
from jax import jit

from .base import Field, GridInfo
from .scalar import ScalarField


class VectorField(Field):
    """3次元ベクトル場"""

    def __init__(
        self,
        grid: GridInfo,
        initial_values: Union[
            Tuple[float, float, float], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ] = (0.0, 0.0, 0.0),
    ):
        """
        ベクトル場を初期化

        Args:
            grid: 計算グリッドの情報
            initial_values: 各成分の初期値
        """
        super().__init__(grid)

        if all(jnp.isscalar(v) for v in initial_values):
            components = tuple(jnp.full(grid.shape, float(v)) for v in initial_values)
        elif all(isinstance(v, jnp.ndarray) for v in initial_values):
            if not all(v.shape == grid.shape for v in initial_values):
                raise ValueError("初期値の形状がグリッドと一致しません")
            components = initial_values
        else:
            raise TypeError("未対応の初期値型")

        self._components = components
        self._scalar_components = [ScalarField(grid, comp) for comp in components]

    @property
    def data(self) -> Tuple[jnp.ndarray, ...]:
        """フィールドデータを取得"""
        return self._components

    @property
    def components(self) -> List[ScalarField]:
        """スカラー成分を取得"""
        return self._scalar_components

    @property
    def ndim(self) -> int:
        """次元数を取得"""
        return len(self._components)

    @jit
    def symmetric_gradient(self) -> "VectorField":
        """対称勾配テンソルを計算

        ∇uₛ = 0.5(∇u + ∇uᵀ)
        """
        # 対称勾配の各成分を計算
        components = []
        for i in range(3):
            for j in range(3):
                # 対称勾配の各成分: 0.5(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
                component_data = 0.5 * (
                    self._scalar_components[i].gradient(j).data
                    + self._scalar_components[j].gradient(i).data
                )
                components.append(ScalarField(self.grid, component_data))

        return VectorField(self.grid, tuple(c.data for c in components[:3]))

    @jit
    def magnitude(self) -> ScalarField:
        """ベクトルの大きさを計算

        Returns:
            大きさを表すスカラー場
        """
        magnitude_array = jnp.sqrt(sum(c * c for c in self._components))
        return ScalarField(self.grid, magnitude_array)

    @jit
    def normalize(self, epsilon: float = 1e-6) -> "VectorField":
        """ベクトル場を正規化

        Args:
            epsilon: 数値安定化のための小さな値

        Returns:
            正規化されたベクトル場
        """
        magnitude = self.magnitude()
        # エピソン追加で0除算を防止
        return self / (magnitude + epsilon)

    @jit
    def dot(self, other: "VectorField") -> ScalarField:
        """内積を計算: v⋅w

        Args:
            other: 内積を取るベクトル場

        Returns:
            内積を表すスカラー場
        """
        if self.grid != other.grid:
            raise ValueError("グリッド情報が一致しません")

        dot_product = sum(
            c1 * c2 for c1, c2 in zip(self._components, other._components)
        )
        return ScalarField(self.grid, dot_product)

    @jit
    def cross(self, other: "VectorField") -> "VectorField":
        """外積を計算: v×w

        Args:
            other: 外積を取るベクトル場

        Returns:
            外積を表すベクトル場
        """
        if self.grid != other.grid:
            raise ValueError("グリッド情報が一致しません")

        # 外積の各成分を計算
        cross_x = (
            self._components[1] * other._components[2]
            - self._components[2] * other._components[1]
        )
        cross_y = (
            self._components[2] * other._components[0]
            - self._components[0] * other._components[2]
        )
        cross_z = (
            self._components[0] * other._components[1]
            - self._components[1] * other._components[0]
        )

        return VectorField(self.grid, (cross_x, cross_y, cross_z))

    @jit
    def divergence(self) -> ScalarField:
        """発散を計算: ∇⋅v

        Returns:
            発散を表すスカラー場
        """
        dx, dy, dz = self.grid.dx
        div = (
            jnp.gradient(self._components[0], axis=0) / dx
            + jnp.gradient(self._components[1], axis=1) / dy
            + jnp.gradient(self._components[2], axis=2) / dz
        )
        return ScalarField(self.grid, div)

    @jit
    def curl(self) -> "VectorField":
        """回転を計算: ∇×v

        Returns:
            回転を表すベクトル場
        """
        dx, dy, dz = self.grid.dx
        # 各成分の偏微分を計算
        dw_dy = jnp.gradient(self._components[2], axis=1) / dy  # ∂w/∂y
        dw_dx = jnp.gradient(self._components[2], axis=0) / dx  # ∂w/∂x
        dv_dx = jnp.gradient(self._components[1], axis=0) / dx  # ∂v/∂x
        dv_dz = jnp.gradient(self._components[1], axis=2) / dz  # ∂v/∂z
        du_dx = jnp.gradient(self._components[0], axis=0) / dx  # ∂u/∂x
        du_dy = jnp.gradient(self._components[0], axis=1) / dy  # ∂u/∂y
        du_dz = jnp.gradient(self._components[0], axis=2) / dz  # ∂u/∂z

        # 回転の各成分を計算: ∇×v = (∂w/∂y - ∂v/∂z, ∂u/∂z - ∂w/∂x, ∂v/∂x - ∂u/∂y)
        curl_x = dw_dy - dv_dz  # ∂w/∂y - ∂v/∂z
        curl_y = du_dz - dw_dx  # ∂u/∂z - ∂w/∂x
        curl_z = dv_dx - du_dy  # ∂v/∂x - ∂u/∂y

        return VectorField(self.grid, (curl_x, curl_y, curl_z))

    def copy(self) -> "VectorField":
        """深いコピーを作成"""
        return VectorField(self.grid, tuple(jnp.array(c) for c in self._components))

    def __add__(self, other: "VectorField") -> "VectorField":
        """加算演算子"""
        if not isinstance(other, VectorField):
            return NotImplemented
        if self.grid != other.grid:
            raise ValueError("グリッド情報が一致しません")

        new_components = tuple(
            c1 + c2 for c1, c2 in zip(self._components, other._components)
        )
        return VectorField(self.grid, new_components)

    def __sub__(self, other: "VectorField") -> "VectorField":
        """減算演算子"""
        if not isinstance(other, VectorField):
            return NotImplemented
        if self.grid != other.grid:
            raise ValueError("グリッド情報が一致しません")

        new_components = tuple(
            c1 - c2 for c1, c2 in zip(self._components, other._components)
        )
        return VectorField(self.grid, new_components)

    def __mul__(
        self, other: Union[float, ScalarField, "VectorField"]
    ) -> Union["VectorField", ScalarField]:
        """乗算演算子

        - スカラー倍: v * a
        - アインシュタインの和による内積: v ⋅ w
        - スカラー場による要素積: v * f
        """
        if jnp.isscalar(other):
            # スカラー倍
            new_components = tuple(c * other for c in self._components)
            return VectorField(self.grid, new_components)
        elif isinstance(other, ScalarField):
            # スカラー場による要素積
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            scalar_array = jnp.asarray(other.data)
            new_components = tuple(c * scalar_array for c in self._components)
            return VectorField(self.grid, new_components)
        elif isinstance(other, VectorField):
            # アインシュタインの和による内積（点積）
            return self.dot(other)
        else:
            return NotImplemented

    def __matmul__(self, other: "VectorField") -> ScalarField:
        """行列積演算子 (@)

        デフォルトで内積と同等の動作
        """
        return self.dot(other)

    def __truediv__(self, other: Union[float, ScalarField]) -> "VectorField":
        """スカラーによる除算"""
        if jnp.isscalar(other):
            new_components = tuple(c / other for c in self._components)
            return VectorField(self.grid, new_components)
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            scalar_array = jnp.asarray(other.data)
            new_components = tuple(c / scalar_array for c in self._components)
            return VectorField(self.grid, new_components)
        else:
            return NotImplemented

    def __pow__(self, power: Union[int, float]) -> "VectorField":
        """累乗演算子

        ベクトル場の各成分に対して累乗を適用
        """
        new_components = tuple(c**power for c in self._components)
        return VectorField(self.grid, new_components)

    def __neg__(self) -> "VectorField":
        """単項マイナス演算子"""
        new_components = tuple(-c for c in self._components)
        return VectorField(self.grid, new_components)

    def __eq__(self, other: "VectorField") -> bool:
        """等価性比較"""
        if not isinstance(other, VectorField):
            return NotImplemented
        if self.grid != other.grid:
            return False
        return all(
            jnp.array_equal(c1, c2)
            for c1, c2 in zip(self._components, other._components)
        )

    def __ne__(self, other: "VectorField") -> bool:
        """非等価性比較"""
        result = self.__eq__(other)
        return not result if result is not NotImplemented else NotImplemented
