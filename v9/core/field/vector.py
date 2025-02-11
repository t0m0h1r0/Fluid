"""ベクトル場クラスを提供するモジュール

このモジュールは、ベクトル量（速度、運動量など）を表現するためのベクトル場クラスを定義します。
"""

from typing import List, Tuple, Optional
import numpy as np
from .scalar import ScalarField
from typing import Dict, Any


class VectorField:
    """ベクトル場クラス

    速度、運動量などのベクトル量を表現するためのクラスです。
    各成分をスカラー場として保持し、ベクトル演算のメソッドを提供します。
    """

    def __init__(self, shape: Tuple[int, ...], dx: float = 1.0):
        """ベクトル場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔
        """
        self._components = [ScalarField(shape, dx) for _ in range(len(shape))]
        self._shape = shape
        self._dx = dx

    @property
    def components(self) -> List[ScalarField]:
        """ベクトル場の各成分を取得"""
        return self._components

    @components.setter
    def components(self, value: List[ScalarField]):
        """ベクトル場の各成分を設定

        Args:
            value: 設定するスカラー場のリスト

        Raises:
            ValueError: コンポーネントの数が不正な場合
        """
        if len(value) != len(self._components):
            raise ValueError(
                f"コンポーネント数が一致しません: {len(value)} != {len(self._components)}"
            )
        if not all(isinstance(v, ScalarField) for v in value):
            raise ValueError("すべての要素がScalarFieldである必要があります")
        if not all(v.shape == self.shape for v in value):
            raise ValueError("すべてのコンポーネントは同じ形状である必要があります")
        self._components = value

    @property
    def shape(self) -> Tuple[int, ...]:
        """場の形状を取得"""
        return self._shape

    @property
    def dx(self) -> float:
        """グリッド間隔を取得"""
        return self._dx

    @property
    def ndim(self) -> int:
        """次元数を取得"""
        return len(self._shape)

    def magnitude(self) -> ScalarField:
        """ベクトル場の大きさを計算

        Returns:
            計算されたベクトルの大きさを表すスカラー場
        """
        result = ScalarField(self.shape, self.dx)
        result.data = np.sqrt(sum(c.data**2 for c in self._components))
        return result

    def divergence(self) -> ScalarField:
        """発散を計算

        Returns:
            計算された発散を表すスカラー場
        """
        result = ScalarField(self.shape, self.dx)
        result.data = sum(c.gradient(i) for i, c in enumerate(self._components))
        return result

    def curl(self) -> Optional["VectorField"]:
        """回転を計算

        Returns:
            3次元の場合は計算された回転を表すベクトル場
            2次元の場合はNone
        """
        if self.ndim != 3:
            return None

        result = VectorField(self.shape, self.dx)
        u, v, w = [c.data for c in self._components]

        # ∂w/∂y - ∂v/∂z
        result.components[0].data = np.gradient(w, self.dx, axis=1) - np.gradient(
            v, self.dx, axis=2
        )
        # ∂u/∂z - ∂w/∂x
        result.components[1].data = np.gradient(u, self.dx, axis=2) - np.gradient(
            w, self.dx, axis=0
        )
        # ∂v/∂x - ∂u/∂y
        result.components[2].data = np.gradient(v, self.dx, axis=0) - np.gradient(
            u, self.dx, axis=1
        )

        return result

    def integrate(self) -> float:
        """場の積分値を計算

        Returns:
            各成分の積分値の二乗和の平方根
        """
        return np.sqrt(sum(c.integrate() ** 2 for c in self._components))

    def copy(self) -> "VectorField":
        """場の深いコピーを作成

        Returns:
            コピーされた場
        """
        result = VectorField(self.shape, self.dx)
        for i, component in enumerate(self._components):
            result.components[i] = component.copy()
        return result

    def __add__(self, other: "VectorField") -> "VectorField":
        """加算演算子の実装"""
        if not isinstance(other, VectorField):
            raise TypeError("ベクトル場同士の演算のみ可能です")
        if self.shape != other.shape:
            raise ValueError("場の形状が一致しません")

        result = VectorField(self.shape, self.dx)
        for i, (c1, c2) in enumerate(zip(self.components, other.components)):
            result.components[i] = c1 + c2
        return result

    def __mul__(self, other) -> "VectorField":
        """スカラー倍の実装"""
        if not isinstance(other, (int, float)):
            raise TypeError("スカラー倍のみ可能です")

        result = VectorField(self.shape, self.dx)
        for i, component in enumerate(self.components):
            result.components[i] = component * other
        return result

    def __rmul__(self, other) -> "VectorField":
        """右スカラー倍の実装"""
        return self.__mul__(other)

    def dot(self, other: "VectorField") -> ScalarField:
        """内積を計算

        Args:
            other: 内積を取るベクトル場

        Returns:
            計算された内積を表すスカラー場
        """
        if not isinstance(other, VectorField):
            raise TypeError("ベクトル場同士の演算のみ可能です")
        if self.shape != other.shape:
            raise ValueError("場の形状が一致しません")

        result = ScalarField(self.shape, self.dx)
        result.data = sum(
            c1.data * c2.data for c1, c2 in zip(self.components, other.components)
        )
        return result

    def cross(self, other: "VectorField") -> Optional["VectorField"]:
        """外積を計算（3次元のみ）

        Args:
            other: 外積を取るベクトル場

        Returns:
            3次元の場合は計算された外積を表すベクトル場
            2次元の場合はNone
        """
        if self.ndim != 3:
            return None
        if not isinstance(other, VectorField):
            raise TypeError("ベクトル場同士の演算のみ可能です")
        if self.shape != other.shape:
            raise ValueError("場の形状が一致しません")

        result = VectorField(self.shape, self.dx)
        u1, v1, w1 = [c.data for c in self.components]
        u2, v2, w2 = [c.data for c in other.components]

        result.components[0].data = v1 * w2 - w1 * v2
        result.components[1].data = w1 * u2 - u1 * w2
        result.components[2].data = u1 * v2 - v1 * u2

        return result

    def save_state(self) -> Dict[str, Any]:
        """現在の状態を保存

        Returns:
            現在の状態を表す辞書
        """
        return {
            "components": [comp.save_state() for comp in self.components],
            "shape": self.shape,
            "dx": self.dx,
        }

    def load_state(self, state: Dict[str, Any]):
        """状態を読み込み

        Args:
            state: 読み込む状態の辞書
        """
        # コンポーネントの復元
        if len(state["components"]) != len(self.components):
            raise ValueError("コンポーネントの数が一致しません")

        for i, comp_state in enumerate(state["components"]):
            self.components[i].load_state(comp_state)

        # その他の属性の確認（必要に応じて）
        if tuple(state["shape"]) != self.shape:
            raise ValueError("形状が一致しません")

        # dx値の確認は省略可（必要に応じて追加）
