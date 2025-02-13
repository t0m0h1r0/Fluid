"""界面の初期化を提供するモジュール

このモジュールは、多相流体計算における界面の初期形状を
生成するための機能を提供します。
"""

from typing import Tuple, List
import numpy as np

from core.field import ScalarField


class InitializationOperator:
    """界面の初期化を実行するクラス"""

    def __init__(self, dx: float):
        """初期化演算子を初期化

        Args:
            dx: グリッド間隔
        """
        self.dx = dx

    def create_sphere(
        self, shape: Tuple[int, ...], center: List[float], radius: float
    ) -> ScalarField:
        """球形の界面を生成

        Args:
            shape: グリッドの形状
            center: 球の中心座標（0-1で正規化）
            radius: 球の半径（0-1で正規化）

        Returns:
            界面からの符号付き距離を表すスカラー場
        """
        result = ScalarField(shape, self.dx)

        # 座標グリッドの生成
        coords = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")

        # 中心からの距離を計算
        squared_distance = sum((x - c) ** 2 for x, c in zip(coords, center))
        distance = np.sqrt(squared_distance)

        # 符号付き距離関数として設定
        result.data = distance - radius
        return result

    def create_plane(
        self, shape: Tuple[int, ...], normal: List[float], point: List[float]
    ) -> ScalarField:
        """平面界面を生成

        Args:
            shape: グリッドの形状
            normal: 平面の法線ベクトル
            point: 平面上の1点の座標（0-1で正規化）

        Returns:
            界面からの符号付き距離を表すスカラー場
        """
        result = ScalarField(shape, self.dx)

        # 座標グリッドの生成
        coords = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")

        # 法線ベクトルの正規化
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)

        # 平面からの符号付き距離を計算
        distance = sum(n * (x - p) for x, p, n in zip(coords, point, normal))

        result.data = distance
        return result

    def create_composite(
        self, phi1: ScalarField, phi2: ScalarField, operation: str = "union"
    ) -> ScalarField:
        """複数の界面を組み合わせて新しい形状を生成

        Args:
            phi1: 1つ目の距離関数
            phi2: 2つ目の距離関数
            operation: 組み合わせ操作（"union", "intersection", "difference"）

        Returns:
            新しい距離関数
        """
        if phi1.shape != phi2.shape:
            raise ValueError("スカラー場の形状が一致しません")

        result = ScalarField(phi1.shape, self.dx)

        if operation == "union":
            # 和集合: min(φ1, φ2)
            result.data = np.minimum(phi1.data, phi2.data)
        elif operation == "intersection":
            # 積集合: max(φ1, φ2)
            result.data = np.maximum(phi1.data, phi2.data)
        elif operation == "difference":
            # 差集合: max(φ1, -φ2)
            result.data = np.maximum(phi1.data, -phi2.data)
        else:
            raise ValueError(f"未知の操作: {operation}")

        return result
