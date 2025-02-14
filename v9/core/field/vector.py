"""ベクトル場クラスを提供するモジュール

このモジュールは、ベクトル量（速度、運動量など）を表現するための
ベクトル場クラスを定義します。"""

from __future__ import annotations  # 型アノテーションの評価を延期
from typing import List, Tuple, Optional, Union, Dict, Any, TYPE_CHECKING
import numpy as np
from .field import Field

if TYPE_CHECKING:
    from .scalar import ScalarField


class VectorField(Field):
    """ベクトル場クラス

    速度、運動量などのベクトル量を表現するためのクラスです。
    各成分をスカラー場として保持し、ベクトル演算のメソッドを提供します。
    """

    def __init__(self, shape: Tuple[int, ...], dx: Union[float, np.ndarray] = 1.0):
        """ベクトル場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔（スカラーまたはベクトル）
        """
        # dxの正規化：スカラーの場合はベクトルに変換
        if np.isscalar(dx):
            dx = np.full(len(shape), float(dx))

        # スーパークラスの初期化
        Field.__init__(self, shape, dx)

        # 循環インポートを避けるために遅延インポート
        from .scalar import ScalarField

        # データの初期化
        self._data = np.zeros(shape + (len(shape),))
        self._components = [ScalarField(shape, dx) for _ in range(len(shape))]

    @property
    def components(self) -> List[ScalarField]:
        """ベクトル場の各成分を取得"""
        # コンポーネントのデータを更新
        for i, comp in enumerate(self._components):
            comp.data = self._data[..., i]
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

        # 各コンポーネントのデータを_dataに反映
        for i, comp in enumerate(value):
            self._data[..., i] = comp.data

        self._components = value

    def magnitude(self) -> ScalarField:
        """ベクトル場の大きさを計算

        Returns:
            計算されたベクトルの大きさを表すスカラー場
        """
        from .scalar import ScalarField
        result = ScalarField(self.shape, self.dx)
        result.data = np.sqrt(sum(c.data**2 for c in self._components))
        return result

    def symmetric_gradient(self) -> VectorField:
        """対称勾配テンソルを計算

        Returns:
            対称勾配テンソルをVectorFieldとして返す
        """
        result = VectorField(self.shape, self.dx)
        for i in range(self.ndim):
            for j in range(self.ndim):
                # 対称勾配: 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
                if i <= j:  # 対称性を利用して計算を最適化
                    comp_data = 0.5 * (
                        self.components[i].gradient(j) + self.components[j].gradient(i)
                    )
                    result.components[i].data = comp_data
                else:
                    # 対称性を利用して既に計算した値をコピー
                    result.components[i].data = result.components[j].data
        return result

    def gradient(self) -> VectorField:
        """ベクトル場の勾配を計算

        Returns:
            勾配テンソルをVectorFieldとして返す
        """
        result = VectorField(self.shape, self.dx)
        for i in range(self.ndim):
            for j in range(self.ndim):
                # ∂u_i/∂x_j
                result.components[i].data = self.components[j].gradient(i)
        return result

    def divergence(self) -> ScalarField:
        """発散を計算

        Returns:
            計算された発散を表すスカラー場
        """
        from .scalar import ScalarField
        result = ScalarField(self.shape, self.dx)
        result.data = sum(c.gradient(i) for i, c in enumerate(self._components))
        return result

    def curl(self) -> Optional[VectorField]:
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
        result.components[0].data = np.gradient(w, self.dx[1], axis=1) - np.gradient(
            v, self.dx[2], axis=2
        )
        # ∂u/∂z - ∂w/∂x
        result.components[1].data = np.gradient(u, self.dx[2], axis=2) - np.gradient(
            w, self.dx[0], axis=0
        )
        # ∂v/∂x - ∂u/∂y
        result.components[2].data = np.gradient(v, self.dx[0], axis=0) - np.gradient(
            u, self.dx[1], axis=1
        )

        return result

    def integrate(self) -> float:
        """場の積分値を計算

        Returns:
            各成分の積分値の二乗和の平方根
        """
        return np.sqrt(sum(c.integrate() ** 2 for c in self._components))

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "max_magnitude": float(self.magnitude().max()),
            "components": self.get_component_diagnostics(),
        }

    def __neg__(self) -> VectorField:
        """単項マイナス演算子の実装

        各成分の符号を反転したVectorFieldを返します。
        """
        result = VectorField(self.shape, self.dx)
        for i, comp in enumerate(self.components):
            result.components[i] = -comp
        return result

    def __add__(self, other: VectorField) -> VectorField:
        """加算演算子の実装"""
        if not isinstance(other, VectorField):
            raise TypeError("ベクトル場同士の演算のみ可能です")
        if self.shape != other.shape:
            raise ValueError("場の形状が一致しません")

        result = VectorField(self.shape, self.dx)
        for i, (c1, c2) in enumerate(zip(self.components, other.components)):
            result.components[i] = c1 + c2
        return result

    def __mul__(self, other: Union[float, ScalarField]) -> VectorField:
        """スカラー倍の実装"""
        result = VectorField(self.shape, self.dx)

        if isinstance(other, (int, float)):
            # スカラー値による乗算
            for i, component in enumerate(self.components):
                result.components[i] = component * other
        elif isinstance(other, ScalarField):
            # スカラー場による乗算
            if self.shape != other.shape:
                raise ValueError("場の形状が一致しません")
            for i, component in enumerate(self.components):
                result.components[i] = component * other
        else:
            raise TypeError("スカラーまたはスカラー場との乗算のみ可能です")

        return result

    def __rmul__(self, other: Union[float, ScalarField]) -> VectorField:
        """右スカラー倍の実装"""
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, ScalarField]) -> VectorField:
        """除算の実装"""
        result = VectorField(self.shape, self.dx)

        if isinstance(other, (int, float)):
            # スカラー値による除算
            for i, component in enumerate(self.components):
                result.components[i] = component / other
        elif isinstance(other, ScalarField):
            # スカラー場による除算
            if self.shape != other.shape:
                raise ValueError("場の形状が一致しません")
            for i, component in enumerate(self.components):
                result.components[i] = component / other
        else:
            raise TypeError("スカラーまたはスカラー場との除算のみ可能です")

        return result

    def dot(self, other: VectorField) -> ScalarField:
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

        from .scalar import ScalarField
        result = ScalarField(self.shape, self.dx)
        result.data = sum(
            c1.data * c2.data for c1, c2 in zip(self.components, other.components)
        )
        return result

    def cross(self, other: VectorField) -> Optional[VectorField]:
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

    def norm(self) -> float:
        """ベクトル場のノルムを計算

        Returns:
            全成分のL2ノルムの最大値
        """
        # 各成分のL2ノルムを計算
        component_norms = [np.sqrt(np.mean(comp.data**2)) for comp in self.components]
        # 最大のノルムを返す
        return max(component_norms)