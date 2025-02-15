"""
3次元ベクトル場を提供するモジュール

このモジュールは、流体力学計算で使用される3次元ベクトル場の実装を提供します。
内部で.baseモジュールのField基底クラスを継承しています。
"""

from __future__ import annotations
from typing import List, Tuple, Union, Sequence
import jax.numpy as np

from .base import Field, FieldFactory, GridInfo
from .scalar import ScalarField


class VectorField(Field):
    """3次元ベクトル場"""

    def __init__(
        self,
        grid: GridInfo,
        initial_values: Sequence[Union[float, np.ndarray, list]] = (0.0, 0.0, 0.0),
    ):
        """ベクトル場を初期化

        Args:
            grid: 計算グリッドの情報
            initial_values: 各成分の初期値。数値、numpy配列、またはリストのシーケンス。
        """
        super().__init__(grid)

        # 入力値の正規化とバリデーション
        components = []
        for value in initial_values:
            if np.isscalar(value):
                # スカラー値の場合
                components.append(FieldFactory.create_scalar_field(grid, float(value)))
            elif isinstance(value, (np.ndarray, list)):
                # 配列またはリストの場合
                data = np.array(value, dtype=np.float64)
                if data.shape != grid.shape:
                    raise ValueError("初期値の形状がグリッドと一致しません")
                components.append(FieldFactory.create_scalar_field(grid, data))
            else:
                raise TypeError(f"未対応の初期値型: {type(value)}")

        # 成分数の確認
        if len(components) != len(grid.shape):
            raise ValueError("成分の数が次元数と一致しません")

        self._components = tuple(components)

    @property
    def data(self) -> Tuple[np.ndarray, ...]:
        """フィールドデータを取得"""
        return tuple(comp.data for comp in self._components)

    @property
    def components(self) -> List[ScalarField]:
        """スカラー成分を取得"""
        return list(self._components)

    def gradient(self, axis: int | None = None) -> Union[VectorField, VectorField]:
        """勾配を計算（各成分のベクトル場）"""
        if axis is not None:
            # 1方向の勾配
            grads = [comp.gradient(axis) for comp in self._components]
            return FieldFactory.create_vector_field(
                self.grid, tuple(g.data for g in grads)
            )
        else:
            # 全方向の勾配（ヤコビアン）
            all_grads = []
            for comp in self._components:
                comp_grads = [comp.gradient(i).data for i in range(len(self.dx))]
                all_grads.extend(comp_grads)
            return FieldFactory.create_vector_field(self.grid, tuple(all_grads))

    def divergence(self) -> ScalarField:
        """発散を計算: ∇⋅v"""
        result = np.zeros(self.shape, dtype=np.float64)
        for i, comp in enumerate(self._components):
            # 各成分の偏微分を中心差分で計算（高次精度）
            result += np.gradient(comp.data, self.dx[i], axis=i, edge_order=2)
        return FieldFactory.create_scalar_field(self.grid, result)

    def curl(self) -> VectorField:
        """回転を計算: ∇×v"""
        # 各方向の偏微分を計算（高次精度）
        derivs = [
            [comp.gradient(j).data for j in range(len(self.dx))]
            for i, comp in enumerate(self._components)
        ]

        # 3次元の場合の回転を計算
        curl_x = derivs[2][1] - derivs[1][2]  # ∂w/∂y - ∂v/∂z
        curl_y = derivs[0][2] - derivs[2][0]  # ∂u/∂z - ∂w/∂x
        curl_z = derivs[1][0] - derivs[0][1]  # ∂v/∂x - ∂u/∂y

        return FieldFactory.create_vector_field(self.grid, (curl_x, curl_y, curl_z))

    def symmetric_gradient(self) -> VectorField:
        """対称勾配テンソルを計算

        ∇ᵤₛ = 0.5(∇u + ∇uᵀ)
        """
        # 対称勾配の各成分を計算
        components = []
        for i in range(3):
            for j in range(3):
                # 対称勾配の各成分: 0.5(∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ)
                grad_i = self._components[i].gradient(j)
                grad_j = self._components[j].gradient(i)
                component = 0.5 * (grad_i + grad_j)
                components.append(component)

        return FieldFactory.create_vector_field(
            self.grid, tuple(comp.data for comp in components[:3])
        )

    def magnitude(self) -> ScalarField:
        """ベクトルの大きさを計算"""
        magnitude_array = np.sqrt(
            sum(comp.data * comp.data for comp in self._components)
        )
        return FieldFactory.create_scalar_field(self.grid, magnitude_array)

    def normalize(self, epsilon: float = 1e-10) -> VectorField:
        """ベクトル場を正規化

        Args:
            epsilon: ゼロ除算を防ぐための小さな値

        Returns:
            正規化されたベクトル場
        """
        magnitude = self.magnitude()
        # エプシロン追加で0除算を防止
        return self / (magnitude + epsilon)

    def dot(self, other: VectorField) -> ScalarField:
        """内積を計算: v⋅w

        Args:
            other: 内積を取る相手のベクトル場

        Returns:
            内積を表すスカラー場
        """
        if self.grid != other.grid:
            raise ValueError("グリッド情報が一致しません")

        dot_product = sum(
            c1.data * c2.data for c1, c2 in zip(self._components, other._components)
        )
        return FieldFactory.create_scalar_field(self.grid, dot_product)

    def cross(self, other: VectorField) -> VectorField:
        """外積を計算: v×w

        Args:
            other: 外積を取る相手のベクトル場

        Returns:
            外積を表すベクトル場
        """
        if self.grid != other.grid:
            raise ValueError("グリッド情報が一致しません")

        # 外積の各成分を計算
        cross_x = (
            self._components[1].data * other._components[2].data
            - self._components[2].data * other._components[1].data
        )
        cross_y = (
            self._components[2].data * other._components[0].data
            - self._components[0].data * other._components[2].data
        )
        cross_z = (
            self._components[0].data * other._components[1].data
            - self._components[1].data * other._components[0].data
        )

        return FieldFactory.create_vector_field(self.grid, (cross_x, cross_y, cross_z))

    def copy(self) -> VectorField:
        """深いコピーを作成"""
        return FieldFactory.create_vector_field(
            self.grid, tuple(np.array(comp.data) for comp in self._components)
        )

    def __add__(self, other: VectorField) -> VectorField:
        """加算演算子"""
        if not isinstance(other, VectorField):
            return NotImplemented
        if self.grid != other.grid:
            raise ValueError("グリッド情報が一致しません")

        new_components = tuple(
            c1.data + c2.data for c1, c2 in zip(self._components, other._components)
        )
        return FieldFactory.create_vector_field(self.grid, new_components)

    def __sub__(self, other: VectorField) -> VectorField:
        """減算演算子"""
        if not isinstance(other, VectorField):
            return NotImplemented
        if self.grid != other.grid:
            raise ValueError("グリッド情報が一致しません")

        new_components = tuple(
            c1.data - c2.data for c1, c2 in zip(self._components, other._components)
        )
        return FieldFactory.create_vector_field(self.grid, new_components)

    def __mul__(self, other: Union[float, ScalarField]) -> VectorField:
        """乗算演算子"""
        if np.isscalar(other):
            new_components = tuple(
                comp.data * float(other) for comp in self._components
            )
            return FieldFactory.create_vector_field(self.grid, new_components)
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            new_components = tuple(comp.data * other.data for comp in self._components)
            return FieldFactory.create_vector_field(self.grid, new_components)
        return NotImplemented

    def __truediv__(self, other: Union[float, ScalarField]) -> VectorField:
        """除算演算子"""
        if np.isscalar(other):
            other_float = float(other)
            if abs(other_float) < np.finfo(np.float64).eps:
                raise ValueError("ゼロ除算は許可されていません")
            new_components = tuple(comp.data / other_float for comp in self._components)
            return FieldFactory.create_vector_field(self.grid, new_components)
        elif isinstance(other, ScalarField):
            if self.grid != other.grid:
                raise ValueError("グリッド情報が一致しません")
            # ゼロ除算を防ぐための微小値を追加
            epsilon = np.finfo(np.float64).eps
            new_components = tuple(
                comp.data / (other.data + epsilon) for comp in self._components
            )
            return FieldFactory.create_vector_field(self.grid, new_components)
        return NotImplemented

    def __neg__(self) -> VectorField:
        """単項マイナス演算子"""
        new_components = tuple(-comp.data for comp in self._components)
        return FieldFactory.create_vector_field(self.grid, new_components)

    def __eq__(self, other: VectorField) -> bool:
        """等価性比較"""
        if not isinstance(other, VectorField):
            return NotImplemented
        if self.grid != other.grid:
            return False
        return all(
            np.array_equal(c1.data, c2.data)
            for c1, c2 in zip(self._components, other._components)
        )
