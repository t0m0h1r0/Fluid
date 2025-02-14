"""スカラー場クラスを提供するモジュール

このモジュールは、スカラー量（圧力、温度など）を表現するための場のクラスを定義します。
"""

from typing import Tuple, Optional, Union, List, TYPE_CHECKING
import numpy as np
from .field import Field
from typing import Dict, Any

if TYPE_CHECKING:
    from .vector import VectorField

class ScalarField(Field):
    """スカラー場クラス

    温度、圧力などのスカラー量を表現するためのクラスです。
    基本的な微分演算や補間機能を提供します。
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dx: Union[float, np.ndarray] = 1.0,
        initial_value: Union[float, np.ndarray] = 0.0,
    ):
        """スカラー場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔（スカラーまたはベクトル）
            initial_value: 初期値（スカラーまたは配列）
        """
        # スーパークラスの初期化
        Field.__init__(self, shape, dx)

        # データの初期化（以前の実装と同じ）
        if isinstance(initial_value, np.ndarray):
            if initial_value.shape != shape:
                raise ValueError(
                    f"Initial value shape {initial_value.shape} does not match field shape {shape}"
                )
            self._data = initial_value.copy()
        elif isinstance(initial_value, (int, float)):
            if initial_value != 0.0:
                self._data.fill(initial_value)
        else:
            raise TypeError(f"Unsupported initial_value type: {type(initial_value)}")

    @property
    def components(self) -> List["ScalarField"]:
        """ScalarFieldは自身を唯一のコンポーネントとして返す"""
        return [self]

    def __add__(self, other: Union["ScalarField", float, int]) -> "ScalarField":
        """加算演算子の実装"""
        if isinstance(other, (float, int)):
            result = self.copy()
            result.data += float(other)
            return result

        if not isinstance(other, ScalarField):
            raise TypeError(f"無効な型との演算: {type(other)}")

        if self.shape != other.shape:
            raise ValueError("ScalarFieldの形状が一致しません")

        result = self.copy()
        result.data += other.data
        return result

    def __mul__(
        self, other: Union["ScalarField", "VectorField", float, int]
    ) -> Union["ScalarField", "VectorField"]:
        """乗算演算子の実装"""
        from .vector import VectorField  # 循環インポート回避

        if isinstance(other, (float, int)):
            result = self.copy()
            result.data *= float(other)
            return result

        if isinstance(other, ScalarField):
            if self.shape != other.shape:
                raise ValueError("ScalarFieldの形状が一致しません")
            result = self.copy()
            result.data *= other.data
            return result

        if isinstance(other, VectorField):
            if self.shape != other.shape:
                raise ValueError("ScalarFieldとVectorFieldの形状が一致しません")
            result = VectorField(self.shape, self.dx)
            for i, comp in enumerate(other.components):
                scalar_result = self.copy()
                scalar_result.data *= comp.data
                result.components[i] = scalar_result
            return result

        raise TypeError(f"無効な型との演算: {type(other)}")

    def __truediv__(self, other: Union["ScalarField", float, int]) -> "ScalarField":
        """除算演算子の実装"""
        if isinstance(other, (float, int)):
            if other == 0:
                raise ZeroDivisionError("ゼロによる除算は許可されません")
            result = self.copy()
            result.data /= float(other)
            return result

        if not isinstance(other, ScalarField):
            raise TypeError(f"無効な型との演算: {type(other)}")

        if self.shape != other.shape:
            raise ValueError("ScalarFieldの形状が一致しません")

        result = self.copy()
        # ゼロ除算の防止
        safe_other_data = np.where(other.data == 0, np.finfo(float).eps, other.data)
        result.data /= safe_other_data
        return result

    def magnitude(self) -> "ScalarField":
        """絶対値の大きさを計算

        Returns:
            絶対値の大きさを表すスカラー場
        """
        result = ScalarField(self.shape, self.dx)
        result.data = np.abs(self.data)
        return result

    def __neg__(self) -> "ScalarField":
        """単項マイナス演算子の実装

        データの符号を反転したScalarFieldを返します。
        """
        result = ScalarField(self.shape, self.dx)
        result.data = -self.data
        return result

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        """任意の点での値を線形補間

        Args:
            points: 補間点の座標 (N, ndim)

        Returns:
            補間された値 (N,)
        """
        # 各次元のインデックスと重みを計算
        indices = []
        weights = []

        for dim in range(self.ndim):
            # 座標をインデックスに変換
            idx = points[:, dim] / self._dx[dim]
            idx0 = np.floor(idx).astype(int)
            idx1 = idx0 + 1
            w1 = idx - idx0
            w0 = 1.0 - w1

            # 境界条件の適用
            idx0 = np.clip(idx0, 0, self.shape[dim] - 1)
            idx1 = np.clip(idx1, 0, self.shape[dim] - 1)

            indices.append((idx0, idx1))
            weights.append((w0, w1))

        # 全ての隣接点での重み付き和を計算
        result = np.zeros(len(points))
        for i in range(2**self.ndim):
            # i のビット表現から各次元でのインデックスを決定
            idx = []
            w = 1.0
            for d in range(self.ndim):
                bit = (i >> d) & 1
                idx.append(indices[d][bit])
                w *= weights[d][bit]

            # インデックスでの値を重み付きで加算
            result += w * self._data[tuple(idx)]

        return result

    def gradient(
        self, axis: Optional[int] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """勾配を計算

        Args:
            axis: 勾配を計算する軸（Noneの場合は全軸の勾配を返す）

        Returns:
            指定された軸の勾配、またはすべての軸の勾配のリスト
        """
        if axis is not None:
            # 特定の軸の勾配を計算
            return np.gradient(self._data, self._dx[axis], axis=axis)

        # すべての軸の勾配を計算
        return [np.gradient(self._data, self._dx[i], axis=i) for i in range(self.ndim)]

    def integrate(self) -> float:
        """場の積分値を計算"""
        return np.sum(self._data) * np.prod(self._dx)

    def mean(self) -> float:
        """場の平均値を計算"""
        return np.mean(self._data)

    def min(self) -> float:
        """場の最小値を取得"""
        return np.min(self._data)

    def max(self) -> float:
        """場の最大値を取得"""
        return np.max(self._data)

    def normalize(self):
        """場を正規化

        場の値を[0, 1]の範囲に正規化します。
        """
        min_val = self.min()
        max_val = self.max()
        if max_val > min_val:
            self._data = (self._data - min_val) / (max_val - min_val)

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        """場の値を指定範囲に制限

        Args:
            min_val: 最小値（Noneの場合は制限なし）
            max_val: 最大値（Noneの場合は制限なし）
        """
        self._data = np.clip(self._data, min_val, max_val)

    def smooth(self, sigma: float = 1.0):
        """場をガウシアンフィルタで平滑化

        Args:
            sigma: ガウシアンフィルタの標準偏差
        """
        from scipy.ndimage import gaussian_filter

        self._data = gaussian_filter(self._data, sigma)

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "magnitude": float(self.magnitude().max()),
            "min": float(self.min()),
            "max": float(self.max()),
            "mean": float(self.mean()),
        }

    def save_state(self) -> Dict[str, Any]:
        """現在の状態を保存

        Returns:
            現在の状態を表す辞書
        """
        return {
            "data": self.data.copy(),
            "shape": self.shape,
            "dx": self.dx,
            "time": self.time,
        }

    def load_state(self, state: Dict[str, Any]):
        """状態を読み込み

        Args:
            state: 読み込む状態の辞書
        """
        # データの復元
        if tuple(state["shape"]) != self.shape:
            raise ValueError("形状が一致しません")

        self.data = state["data"].copy()

        # 時刻の復元
        self.time = state.get("time", 0.0)

        # グリッド間隔の確認（必要に応じて）
        if not np.allclose(state["dx"], self.dx):
            raise ValueError("グリッド間隔が一致しません")

    def norm(self, ord=2) -> float:
        """場のノルムを計算

        Args:
            ord: ノルムの種類（デフォルトはL2ノルム）

        Returns:
            計算されたノルム
        """
        return np.linalg.norm(self._data.ravel(), ord=ord)
