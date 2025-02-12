"""Level Set関数の初期化を提供するモジュール"""

from typing import Tuple, Dict, Any
import numpy as np

from .base import BaseLevelSetOperation


class LevelSetInitializer(BaseLevelSetOperation):
    """Level Set関数の初期化クラス"""

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
        # グリッドの生成（インデックス用）
        self.coords = np.meshgrid(
            *[np.linspace(0, 1, s) for s in shape], indexing="ij"
        )
        
        # 未設定部分を大きな正の値で初期化
        phi = np.full(shape, np.inf)

        # オブジェクトリストと背景相の取得
        objects = kwargs.get("objects", [])
        background_phase = kwargs.get("background_phase", "nitrogen")

        # 各オブジェクトの処理
        for obj in objects:
            obj_type = obj.get("type")
            obj_phase = obj.get("phase", "water")

            # レベルセット関数の計算
            if obj_type == "plate":
                phi_new = self._compute_plate_levelset(obj)
            elif obj_type == "sphere":
                phi_new = self._compute_sphere_levelset(obj)
            else:
                continue

            # 相に応じたレベルセット関数の更新
            if obj_phase != background_phase:
                # 異なる相のオブジェクトならminを取る（負の値が内部を示す）
                phi = np.minimum(phi, -phi_new)
            else:
                # 背景相と同じ相ならmaxを取る（正の値が外部を示す）
                phi = np.maximum(phi, phi_new)

        return phi

    def _compute_sphere_levelset(self, obj: Dict[str, Any]) -> np.ndarray:
        """球のレベルセット関数を計算

        Args:
            obj: 球オブジェクトの設定
                - center: 球の中心座標
                - radius: 球の半径

        Returns:
            計算されたレベルセット関数の値
        """
        # デフォルト値の設定
        center = obj.get("center", [0.5, 0.5, 0.5][:len(self.coords)])
        radius = obj.get("radius", 0.2)

        # 中心からの距離を計算
        squared_distance = sum(
            (coord - cent) ** 2 
            for coord, cent in zip(self.coords[:len(center)], center)
        )
        
        # 符号付き距離関数を計算
        distance = np.sqrt(squared_distance)
        return distance - radius

    def _compute_plate_levelset(self, obj: Dict[str, Any]) -> np.ndarray:
        """平面のレベルセット関数を計算

        Args:
            obj: 平面オブジェクトの設定
                - height: 平面の高さ（0-1の範囲）

        Returns:
            計算されたレベルセット関数の値
        """
        # 高さの設定とバリデーション
        height = obj.get("height", 0.5)
        if not 0 <= height <= 1:
            raise ValueError("高さは0から1の間である必要があります")
        
        # Z座標との差を計算（正: 平面より上、負: 平面より下）
        return self.coords[-1] - height

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