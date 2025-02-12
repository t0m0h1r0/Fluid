"""Level Set法の中心的なデータ構造を提供するモジュール

このモジュールは、Level Set法における主要なデータ構造である
LevelSetFieldクラスを定義します。
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np

from core.field import ScalarField, VectorField
from .geometry.curvature import CurvatureCalculator
from .geometry.normal import NormalCalculator
from .indicator import HeavisideFunction, DeltaFunction
from .operations.initializer import LevelSetInitializer
from .operations.reinitializer import LevelSetReinitializer
from .utils.validators import validate_shape, validate_dx


@dataclass
class LevelSetParameters:
    """Level Set法のパラメータを管理するデータクラス"""

    epsilon: float = 1.0e-2  # 界面の厚さ
    reinit_interval: int = 5  # 再初期化の間隔
    reinit_steps: int = 2  # 再初期化のステップ数
    min_value: float = 1.0e-10  # 最小値の閾値


class LevelSetField:
    """Level Set関数を表現するクラス"""

    def __init__(
        self,
        shape: Tuple[int, ...],
        dx: float = 1.0,
        params: Optional[LevelSetParameters] = None,
    ):
        """Level Set関数を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔
            params: Level Set法のパラメータ
        """
        validate_shape(shape)
        validate_dx(dx)

        self._data = np.zeros(shape)
        self._dx = dx
        self.params = params or LevelSetParameters()

        # 各種計算機のインスタンス化
        self._heaviside = HeavisideFunction(self.params.epsilon)
        self._delta = DeltaFunction(self.params.epsilon)
        self._curvature = CurvatureCalculator(dx)
        self._normal = NormalCalculator(dx)
        self._initializer = LevelSetInitializer(dx)
        self._reinitializer = LevelSetReinitializer(dx, self.params.epsilon)

    @property
    def data(self) -> np.ndarray:
        """Level Set関数のデータを取得"""
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Level Set関数のデータを設定"""
        if value.shape != self._data.shape:
            raise ValueError(f"無効な形状: {value.shape} != {self._data.shape}")
        self._data = value

    @property
    def shape(self) -> Tuple[int, ...]:
        """グリッドの形状を取得"""
        return self._data.shape

    @property
    def dx(self) -> float:
        """グリッド間隔を取得"""
        return self._dx

    def initialize(self, **kwargs) -> None:
        """Level Set関数を初期化"""
        self._data = self._initializer.initialize(self.shape, **kwargs)

    def reinitialize(self) -> None:
        """Level Set関数を再初期化"""
        self._data = self._reinitializer.reinitialize(
            self._data, n_steps=self.params.reinit_steps
        )

    def get_heaviside(self) -> ScalarField:
        """Heaviside関数を計算"""
        result = ScalarField(self.shape, self.dx)
        result.data = self._heaviside.compute(self._data)
        return result

    def get_delta(self) -> ScalarField:
        """Delta関数を計算"""
        result = ScalarField(self.shape, self.dx)
        result.data = self._delta.compute(self._data)
        return result

    def get_curvature(self, method: str = "standard") -> ScalarField:
        """界面の曲率を計算

        Args:
            method: 計算手法 ('standard' または 'high_order')

        Returns:
            曲率を表すScalarField
        """
        result = ScalarField(self.shape, self.dx)
        result.data = self._curvature.compute(self._data, method=method)
        return result

    def get_normal(self) -> VectorField:
        """界面の法線ベクトルを計算

        Returns:
            法線ベクトルを表すVectorField
        """
        return self._normal.compute(self._data)

    def get_density(self, rho1: float, rho2: float) -> ScalarField:
        """密度場を計算: ρ1H(ϕ) + ρ2(1-H(ϕ))

        Args:
            rho1: 第1相の密度
            rho2: 第2相の密度

        Returns:
            密度を表すScalarField
        """
        result = ScalarField(self.shape, self.dx)
        h = self._heaviside.compute(self._data)
        result.data = rho1 * h + rho2 * (1.0 - h)
        return result

    def get_volume(self) -> float:
        """体積を計算"""
        h = self._heaviside.compute(self._data)
        return float(np.sum(h) * self._dx ** len(self.shape))

    def get_area(self) -> float:
        """界面の面積を計算"""
        d = self._delta.compute(self._data)
        return float(np.sum(d) * self._dx ** len(self.shape))

    def gradient(self, axis: int) -> np.ndarray:
        """指定軸方向の勾配を計算"""
        return np.gradient(self._data, self._dx, axis=axis)

    def copy(self) -> "LevelSetField":
        """深いコピーを作成"""
        new_field = self.__class__(shape=self.shape, dx=self.dx, params=self.params)
        new_field.data = self._data.copy()
        return new_field

    def get_geometry_info(self) -> Dict[str, Any]:
        """幾何学的な情報を取得"""
        kappa = self.get_curvature()
        return {
            "volume": self.get_volume(),
            "area": self.get_area(),
            "min_curvature": float(np.min(kappa.data)),
            "max_curvature": float(np.max(kappa.data)),
            "mean_curvature": float(np.mean(kappa.data)),
        }
