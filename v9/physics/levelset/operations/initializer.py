"""Level Set関数の初期化を提供するモジュール"""

from typing import Tuple
import numpy as np

from .base import BaseLevelSetOperation


class LevelSetInitializer(BaseLevelSetOperation):
    """Level Set関数の初期化クラス"""

    def initialize(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        """Level Set関数を初期化

        Args:
            shape: グリッドの形状
            **kwargs: 初期化パラメータ
                - method: 初期化方法 ("sphere", "box", "plane")
                - center: 中心座標 (球体/箱用)
                - radius: 半径 (球体用)
                - normal: 法線ベクトル (平面用)
                - point: 平面上の点 (平面用)

        Returns:
            初期化されたLevel Set関数の値
        """
        method = kwargs.get("method", "sphere")

        if method == "sphere":
            return self._initialize_sphere(shape, **kwargs)
        elif method == "box":
            return self._initialize_box(shape, **kwargs)
        elif method == "plane":
            return self._initialize_plane(shape, **kwargs)
        else:
            raise ValueError(f"未知の初期化方法: {method}")

    def _initialize_sphere(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        """球状の初期化

        Args:
            shape: グリッドの形状
            **kwargs:
                - center: 中心座標
                - radius: 半径

        Returns:
            初期化されたLevel Set関数の値
        """
        center = kwargs.get("center") or [s / 2 for s in shape]
        radius = kwargs.get("radius") or min(shape) / 4

        # 座標グリッドの生成
        coords = np.ogrid[tuple(slice(0, s) for s in shape)]
        coords = [c - cent for c, cent in zip(coords, center)]

        # 各点から中心までの距離を計算
        distance = np.sqrt(sum(c**2 for c in coords))

        # 符号付き距離関数を計算
        return distance - radius

    def _initialize_box(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        """箱状の初期化

        Args:
            shape: グリッドの形状
            **kwargs:
                - center: 中心座標
                - size: 箱のサイズ

        Returns:
            初期化されたLevel Set関数の値
        """
        center = kwargs.get("center") or [s / 2 for s in shape]
        size = kwargs.get("size") or [s / 4 for s in shape]

        # 座標グリッドの生成
        coords = np.ogrid[tuple(slice(0, s) for s in shape)]
        coords = [abs(c - cent) - sz / 2 for c, cent, sz in zip(coords, center, size)]

        # 符号付き距離関数を計算
        return np.maximum(0, np.maximum(*coords))

    def _initialize_plane(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        """平面での初期化

        Args:
            shape: グリッドの形状
            **kwargs:
                - normal: 法線ベクトル
                - point: 平面上の点

        Returns:
            初期化されたLevel Set関数の値
        """
        normal = kwargs.get("normal") or [0, 0, 1]
        point = kwargs.get("point") or [s / 2 for s in shape]

        # 法線ベクトルの正規化
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)

        # 座標グリッドの生成
        coords = np.ogrid[tuple(slice(0, s) for s in shape)]
        coords = [c - p for c, p in zip(coords, point)]

        # 平面までの符号付き距離を計算
        return sum(n * c for n, c in zip(normal, coords))

    def validate_input(self, phi: np.ndarray) -> None:
        """入力データを検証

        Args:
            phi: Level Set関数の値
        """
        if not isinstance(phi, np.ndarray):
            raise TypeError("入力はnumpy配列である必要があります")
        if phi.ndim not in [2, 3]:
            raise ValueError("2次元または3次元の配列である必要があります")
