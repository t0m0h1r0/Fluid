"""Level Set関数の初期化を提供するモジュール"""

from typing import Tuple, Dict, Any
import numpy as np

from .base import BaseLevelSetOperation


class LevelSetInitializer(BaseLevelSetOperation):
    """Level Set関数の初期化クラス"""

class LevelSetInitializer(BaseLevelSetOperation):
    def initialize(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        """Level Set関数を初期化

        Args:
            shape: グリッドの形状
            **kwargs: 初期化パラメータ
                - objects: オブジェクトのリスト
                - background_phase: 背景の相
                
        Returns:
            初期化されたLevel Set関数の値
        """
        # グリッドの生成（実際の座標用）
        self.coords = np.meshgrid(
            *[np.linspace(0, 1, s) for s in shape], indexing="ij"
        )

        # 背景相の初期化
        phi = np.full(shape, 1e6)  # np.infを避ける

        # オブジェクトリストの取得
        objects = kwargs.get("objects", [])

        # 各オブジェクトの処理
        for obj in objects:
            obj_type = obj.get("type")

            # レベルセット関数の計算
            if obj_type == "plate":
                phi_new = self._compute_plate_levelset(obj, shape)
            elif obj_type == "sphere":
                phi_new = self._compute_sphere_levelset(obj, shape)
            else:
                continue

            # 最小値を取ることで、各オブジェクトの内外が正しく設定される
            phi = np.minimum(phi, phi_new)

        return phi

    def _compute_sphere_levelset(self, obj: Dict[str, Any], shape: Tuple[int, ...]) -> np.ndarray:
        """球のレベルセット関数を計算

        Args:
            obj: 球オブジェクトの設定
            shape: グリッドの形状

        Returns:
            計算されたレベルセット関数の値
        """
        # デフォルト値の設定
        center = obj.get("center", [0.5, 0.5, 0.5][:len(shape)])
        radius = obj.get("radius", 0.2)

        # 中心からの距離を計算
        squared_distance = sum(
            (coord - cent) ** 2 
            for coord, cent in zip(self.coords[:len(center)], center)
        )
        
        # 符号付き距離関数を計算
        distance = np.sqrt(squared_distance)
        return distance - radius

    def _compute_plate_levelset(self, obj: Dict[str, Any], shape: Tuple[int, ...]) -> np.ndarray:
        """平面のレベルセット関数を計算

        Args:
            obj: 平面オブジェクトの設定
            shape: グリッドの形状

        Returns:
            計算されたレベルセット関数の値
        """
        # 高さの設定とバリデーション
        height = obj.get("height", 0.5)
        if not 0 <= height <= 1:
            raise ValueError("高さは0から1の間である必要があります")
        
        # 全次元で平面を表現
        # Z座標との差を計算（正: 平面より上、負: 平面より下）
        plane = np.zeros(shape)
        for coord in self.coords:
            plane += np.abs(coord - height)
        
        return plane

    def validate_input(self, phi: np.ndarray) -> None:
        """入力データを検証

        Args:
            phi: Level Set関数の値

        Raises:
            TypeError: 入力が不適切な型の場合
            ValueError: 入力の次元が不適切な場合
        """
        if not isinstance(phi, np.ndarray):
            raise TypeError("入力はnumpy配列である必要があります")
        if phi.ndim < 1 or phi.ndim > 3:
            raise ValueError("1次元, 2次元または3次元の配列である必要があります")