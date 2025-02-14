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

        # 基底クラスの初期化
        super().__init__(shape, dx)

        # 循環インポートを避けるために遅延インポート
        from .scalar import ScalarField

        # データの初期化
        self._data = np.zeros(shape + (len(shape),))
        self._components = [
            ScalarField(shape, dx, initial_value=self._data[..., i].copy())
            for i in range(len(shape))
        ]

    @property
    def components(self) -> List[ScalarField]:
        """ベクトル場の各成分を取得"""
        # 各コンポーネントのデータを更新
        for i, comp in enumerate(self._components):
            comp._data = self._data[..., i].copy()  # 直接_dataを更新
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
        if not all(v.shape == self.shape[:-1] for v in value):
            raise ValueError("すべてのコンポーネントは同じ形状である必要があります")

        # 各コンポーネントのデータを更新
        for i, comp in enumerate(value):
            self._data[..., i] = comp.data
            self._components[i] = comp

    def magnitude(self) -> ScalarField:
        """ベクトル場の大きさを計算

        Returns:
            計算されたベクトルの大きさを表すスカラー場
        """
        from .scalar import ScalarField

        result = ScalarField(self.shape[:-1], self.dx)  # 形状から最後の次元を除外
        result.data = np.sqrt(np.sum(self._data**2, axis=-1))
        return result

    def symmetric_gradient(self) -> VectorField:
        """対称勾配テンソルを計算

        Returns:
            対称勾配テンソルをVectorFieldとして返す
        """
        result = VectorField(self.shape[:-1], self.dx)  # 形状から最後の次元を除外
        for i in range(self.ndim):
            for j in range(self.ndim):
                # 対称勾配: 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
                if i <= j:  # 対称性を利用して計算を最適化
                    comp_data = 0.5 * (
                        np.gradient(self._data[..., i], self.dx[j], axis=j)
                        + np.gradient(self._data[..., j], self.dx[i], axis=i)
                    )
                    result._data[..., i] = comp_data
                else:
                    # 対称性を利用して既に計算した値をコピー
                    result._data[..., i] = result._data[..., j]
        return result

    def gradient(self) -> VectorField:
        """ベクトル場の勾配を計算

        Returns:
            勾配テンソルをVectorFieldとして返す
        """
        result = VectorField(self.shape[:-1], self.dx)  # 形状から最後の次元を除外
        for i in range(self.ndim):
            for j in range(self.ndim):
                # ∂u_i/∂x_j
                result._data[..., i] = np.gradient(
                    self._data[..., j], self.dx[i], axis=i
                )
        return result

    def divergence(self) -> ScalarField:
        """発散を計算

        Returns:
            計算された発散を表すスカラー場
        """
        from .scalar import ScalarField

        result = ScalarField(self.shape[:-1], self.dx)  # 形状から最後の次元を除外
        divs = [
            np.gradient(self._data[..., i], self.dx[i], axis=i)
            for i in range(self.ndim)
        ]
        result.data = np.sum(divs, axis=0)
        return result

    def curl(self) -> Optional[VectorField]:
        """回転を計算（3次元のみ）

        Returns:
            3次元の場合は計算された回転を表すベクトル場
            2次元の場合はNone
        """
        if self.ndim != 3:
            return None

        result = VectorField(self.shape[:-1], self.dx)  # 形状から最後の次元を除外
        # ∂w/∂y - ∂v/∂z
        result._data[..., 0] = np.gradient(
            self._data[..., 2], self.dx[1], axis=1
        ) - np.gradient(self._data[..., 1], self.dx[2], axis=2)
        # ∂u/∂z - ∂w/∂x
        result._data[..., 1] = np.gradient(
            self._data[..., 0], self.dx[2], axis=2
        ) - np.gradient(self._data[..., 2], self.dx[0], axis=0)
        # ∂v/∂x - ∂u/∂y
        result._data[..., 2] = np.gradient(
            self._data[..., 1], self.dx[0], axis=0
        ) - np.gradient(self._data[..., 0], self.dx[1], axis=1)
        return result

    def integrate(self) -> float:
        """場の積分値を計算

        Returns:
            各成分の積分値の二乗和の平方根
        """
        # グリッド体積要素
        dV = np.prod(self.dx)
        # 各成分の積分
        integral_squares = [
            np.sum(self._data[..., i]) ** 2 * dV**2 for i in range(self.ndim)
        ]
        return np.sqrt(np.sum(integral_squares))

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "max_magnitude": float(self.magnitude().max()),
            "components": {
                f"component_{i}": {
                    "min": float(np.min(self._data[..., i])),
                    "max": float(np.max(self._data[..., i])),
                    "mean": float(np.mean(self._data[..., i])),
                    "norm": float(np.linalg.norm(self._data[..., i])),
                }
                for i in range(self.ndim)
            },
        }

    def __neg__(self) -> VectorField:
        """単項マイナス演算子の実装"""
        result = VectorField(self.shape[:-1], self.dx)
        result._data = -self._data
        return result

    def __add__(self, other: VectorField) -> VectorField:
        """加算演算子の実装"""
        if not isinstance(other, VectorField):
            raise TypeError("ベクトル場同士の演算のみ可能です")
        if self.shape != other.shape:
            raise ValueError("場の形状が一致しません")

        result = VectorField(self.shape[:-1], self.dx)
        result._data = self._data + other._data
        return result

    def __mul__(self, other: Union[float, ScalarField]) -> VectorField:
        """スカラー倍の実装"""
        result = VectorField(self.shape[:-1], self.dx)

        if isinstance(other, (int, float)):
            result._data = self._data * other
        elif isinstance(other, ScalarField):
            if other.shape != self.shape[:-1]:
                raise ValueError("場の形状が一致しません")
            # スカラー場を各成分に掛ける
            result._data = self._data * other.data[..., np.newaxis]
        else:
            raise TypeError("スカラーまたはスカラー場との乗算のみ可能です")

        return result

    def __rmul__(self, other: Union[float, ScalarField]) -> VectorField:
        """右スカラー倍の実装"""
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, ScalarField]) -> VectorField:
        """除算の実装"""
        result = VectorField(self.shape[:-1], self.dx)

        if isinstance(other, (int, float)):
            result._data = self._data / other
        elif isinstance(other, ScalarField):
            if other.shape != self.shape[:-1]:
                raise ValueError("場の形状が一致しません")
            # スカラー場で各成分を割る
            result._data = self._data / other.data[..., np.newaxis]
        else:
            raise TypeError("スカラーまたはスカラー場との除算のみ可能です")

        return result

    def dot(self, other: VectorField) -> ScalarField:
        """内積を計算"""
        if not isinstance(other, VectorField):
            raise TypeError("ベクトル場同士の演算のみ可能です")
        if self.shape != other.shape:
            raise ValueError("場の形状が一致しません")

        from .scalar import ScalarField

        result = ScalarField(self.shape[:-1], self.dx)
        result.data = np.sum(self._data * other._data, axis=-1)
        return result

    def cross(self, other: VectorField) -> Optional[VectorField]:
        """外積を計算（3次元のみ）"""
        if self.ndim != 3:
            return None
        if not isinstance(other, VectorField):
            raise TypeError("ベクトル場同士の演算のみ可能です")
        if self.shape != other.shape:
            raise ValueError("場の形状が一致しません")

        result = VectorField(self.shape[:-1], self.dx)
        # 外積の各成分を計算
        result._data[..., 0] = (
            self._data[..., 1] * other._data[..., 2]
            - self._data[..., 2] * other._data[..., 1]
        )
        result._data[..., 1] = (
            self._data[..., 2] * other._data[..., 0]
            - self._data[..., 0] * other._data[..., 2]
        )
        result._data[..., 2] = (
            self._data[..., 0] * other._data[..., 1]
            - self._data[..., 1] * other._data[..., 0]
        )
        return result

    def norm(self) -> float:
        """ベクトル場のノルムを計算

        Returns:
            全成分のL2ノルムの最大値
        """
        # 各成分のL2ノルムを計算
        component_norms = [
            np.sqrt(np.mean(self._data[..., i] ** 2)) for i in range(self.ndim)
        ]
        # 最大のノルムを返す
        return max(component_norms)
