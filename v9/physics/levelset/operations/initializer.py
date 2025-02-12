"""Level Set関数の初期化を提供するモジュール"""

from typing import Tuple, Dict, Any, List
import numpy as np

from .base import BaseLevelSetOperation


class LevelSetInitializer(BaseLevelSetOperation):
    """Level Set関数の高度な初期化クラス"""

    def initialize(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        """
        Level Set関数を初期化

        Args:
            shape: グリッドの形状
            **kwargs: 初期化パラメータ
                - objects: オブジェクトのリスト
                - background_phase: 背景の相
                
        Returns:
            初期化されたLevel Set関数の値
        """
        # グリッドの生成
        coords = np.meshgrid(
            *[np.linspace(0, 1, s) for s in shape], indexing="ij"
        )
        
        # 初期値を設定
        phi = np.full(shape, np.inf)

        # オブジェクトリストと背景相の取得
        objects = kwargs.get("objects", [])
        background_phase = kwargs.get("background_phase", "water")

        # オブジェクトの優先順位リストを作成
        prioritized_objects = self._prioritize_objects(objects, background_phase)

        # 各オブジェクトの処理
        for obj in prioritized_objects:
            phi_obj = self._compute_object_levelset(obj, shape)
            
            # オブジェクトの相に応じて更新戦略を選択
            if obj['phase'].lower() != background_phase.lower():
                # 異なる相のオブジェクト：最小値を取る（界面を保持）
                phi = np.minimum(phi, phi_obj)
            else:
                # 背景相と同じ相のオブジェクト：最大値を取る
                phi = np.maximum(phi, -phi_obj)

        return phi

    def _prioritize_objects(
        self, 
        objects: List[Dict[str, Any]], 
        background_phase: str
    ) -> List[Dict[str, Any]]:
        """
        オブジェクトの優先順位を決定

        Args:
            objects: オブジェクトのリスト
            background_phase: 背景相

        Returns:
            優先順位付きのオブジェクトリスト
        """
        def priority_key(obj):
            # 背景相と異なる相のオブジェクトを優先
            is_different_phase = obj.get('phase', '').lower() != background_phase.lower()
            
            # オブジェクトタイプに基づく追加の優先順位
            type_priority = {
                'sphere': 10,    # 球体を最優先
                'plate': 5,      # 平面を次点
                'background': 1  # 背景オブジェクトは最後
            }
            
            type_score = type_priority.get(obj.get('type', 'background'), 0)
            
            return (-is_different_phase, -type_score)

        return sorted(objects, key=priority_key)

    def _compute_object_levelset(
        self, 
        obj: Dict[str, Any], 
        shape: Tuple[int, ...]
    ) -> np.ndarray:
        """
        特定のオブジェクトのレベルセット関数を計算

        Args:
            obj: オブジェクトの設定
            shape: グリッドの形状

        Returns:
            オブジェクトのレベルセット関数
        """
        # グリッドの生成
        coords = np.meshgrid(
            *[np.linspace(0, 1, s) for s in shape], indexing="ij"
        )

        obj_type = obj.get('type')

        if obj_type == 'plate':
            height = obj.get('height', 0.5)
            if not 0 <= height <= 1:
                raise ValueError("高さは0から1の間である必要があります")
            return coords[-1] - height

        elif obj_type == 'sphere':
            center = obj.get('center', [0.5, 0.5, 0.5])
            radius = obj.get('radius', 0.2)

            # 中心からの距離を計算
            squared_distance = sum(
                (coord - c) ** 2 for coord, c in zip(coords, center)
            )
            distance = np.sqrt(squared_distance)
            
            # 符号付き距離関数を計算
            return distance - radius

        elif obj_type == 'background':
            # デフォルトの背景（全空間が同一相）
            return np.zeros_like(coords[0])

        else:
            raise ValueError(f"サポートされていないオブジェクトタイプ: {obj_type}")

    def validate_input(self, phi: np.ndarray) -> None:
        """入力データを検証"""
        if not isinstance(phi, np.ndarray):
            raise TypeError("入力はnumpy配列である必要があります")
        if phi.ndim < 1 or phi.ndim > 3:
            raise ValueError("1次元, 2次元または3次元の配列である必要があります")