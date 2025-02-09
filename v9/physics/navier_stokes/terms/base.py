"""Navier-Stokes方程式の項の基底クラスを提供するモジュール"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from core.field import VectorField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager


class NavierStokesTerm(ABC):
    """Navier-Stokes方程式の項の基底クラス"""

    @property
    @abstractmethod
    def name(self) -> str:
        """項の名前を取得"""
        pass

    @abstractmethod
    def compute(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> List[np.ndarray]:
        """項の寄与を計算

        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            **kwargs: 追加のパラメータ

        Returns:
            各方向の速度成分への寄与のリスト
        """
        pass

    @abstractmethod
    def get_diagnostics(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> Dict[str, Any]:
        """診断情報を取得

        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            **kwargs: 追加のパラメータ

        Returns:
            診断情報を含む辞書
        """
        pass
