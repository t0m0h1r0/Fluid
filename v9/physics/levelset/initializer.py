"""Level Set関数の初期化を担当するモジュール"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import numpy as np

from .field import LevelSetField, LevelSetParameters


class Phase(Enum):
    """流体の相を表す列挙型"""
    PHASE_1 = 1  # 第1相（例：水）
    PHASE_2 = 2  # 第2相（例：空気）


@dataclass
class InterfaceObject:
    """界面オブジェクトを表すデータクラス"""
    phase: Phase
    object_type: str  # "background", "layer", "sphere"
    height: Optional[float] = None  # レイヤー用
    center: Optional[Tuple[float, float, float]] = None  # 球体用
    radius: Optional[float] = None  # 球体用


class LevelSetInitializer:
    """Level Set関数の初期化を担当するクラス"""

    def __init__(self, dx: float, epsilon: float = None):
        """初期化子を構築
        
        Args:
            dx: グリッド間隔
            epsilon: 界面の厚さ（指定がない場合はdxから自動設定）
        """
        self.dx = dx
        self.epsilon = epsilon or (1.5 * dx)

    def initialize(
        self,
        shape: Tuple[int, ...],
        objects: List[InterfaceObject]
    ) -> LevelSetField:
        """Level Set関数を初期化
        
        Args:
            shape: グリッドの形状
            objects: 界面オブジェクトのリスト

        Returns:
            初期化されたLevel Set場
        """
        # Level Setパラメータの設定
        params = LevelSetParameters(
            epsilon=self.epsilon,
            reinit_interval=5,
            reinit_steps=2
        )

        # Level Set場の作成
        levelset = LevelSetField(shape=shape, dx=self.dx, params=params)

        # 背景相の設定
        background = next(obj for obj in objects if obj.object_type == "background")
        self._set_background(levelset, background.phase)

        # その他のオブジェクトを適用
        for obj in objects:
            if obj.object_type != "background":
                self._apply_object(levelset, obj)

        return levelset

    def _set_background(self, levelset: LevelSetField, phase: Phase) -> None:
        """背景相を設定"""
        # PHASE_1なら負の値（内部）、PHASE_2なら正の値（外部）で初期化
        levelset.data.fill(-1.0 if phase == Phase.PHASE_1 else 1.0)

    def _apply_object(self, levelset: LevelSetField, obj: InterfaceObject) -> None:
        """オブジェクトを適用"""
        if obj.object_type == "layer":
            self._apply_layer(levelset, obj)
        elif obj.object_type == "sphere":
            self._apply_sphere(levelset, obj)

    def _apply_layer(self, levelset: LevelSetField, obj: InterfaceObject) -> None:
        """水平レイヤーを適用"""
        if obj.height is None:
            raise ValueError("レイヤーには高さの指定が必要です")

        # 座標グリッドを作成
        coords = np.meshgrid(
            *[np.arange(n) * self.dx for n in levelset.shape],
            indexing='ij'
        )
        
        # 高さ方向の座標（最後の次元）
        z = coords[-1]
        
        # 符号付き距離関数を計算
        phi = obj.height - z
        
        # 相に応じて符号を反転
        if obj.phase == Phase.PHASE_2:
            phi = -phi
            
        # Level Set関数を更新（CSG演算）
        levelset.data = np.minimum(levelset.data, phi)

    def _apply_sphere(self, levelset: LevelSetField, obj: InterfaceObject) -> None:
        """球体を適用"""
        if obj.center is None or obj.radius is None:
            raise ValueError("球体には中心座標と半径の指定が必要です")

        # 座標グリッドを作成
        coords = np.meshgrid(
            *[np.arange(n) * self.dx for n in levelset.shape],
            indexing='ij'
        )
        
        # 球の中心からの距離を計算
        squared_distance = sum(
            (coord - center) ** 2 
            for coord, center in zip(coords, obj.center)
        )
        distance = np.sqrt(squared_distance)
        
        # 符号付き距離関数を計算
        phi = obj.radius - distance
        
        # 相に応じて符号を反転
        if obj.phase == Phase.PHASE_2:
            phi = -phi
            
        # Level Set関数を更新（CSG演算）
        levelset.data = np.minimum(levelset.data, phi)