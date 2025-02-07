"""Navier-Stokes方程式の各項の基底クラスを提供するモジュール

このモジュールは、Navier-Stokes方程式の各項（移流項、粘性項、圧力項、外力項）
の基底となる抽象クラスを定義します。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from core.field import VectorField


class NavierStokesTerm(ABC):
    """Navier-Stokes方程式の項の基底クラス

    このクラスは、Navier-Stokes方程式の各項に共通のインターフェースを定義します。
    各項は、速度場の時間変化への寄与を計算します。
    """

    def __init__(self, name: str):
        """基底クラスを初期化

        Args:
            name: 項の名前
        """
        self.name = name
        self._enabled = True  # 項を有効/無効化するフラグ

    @property
    def enabled(self) -> bool:
        """項が有効かどうかを取得"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """項の有効/無効を設定"""
        self._enabled = value

    @abstractmethod
    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        """項の寄与を計算

        Args:
            velocity: 現在の速度場
            **kwargs: 追加のパラメータ（圧力場、物性値など）

        Returns:
            各方向の速度成分への寄与のリスト
        """
        pass

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 現在の速度場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅の制限
            デフォルトではfloat('inf')を返し、制限なしを表す
        """
        return float("inf")

    def initialize(self, **kwargs) -> None:
        """項の初期化処理

        Args:
            **kwargs: 初期化に必要なパラメータ
        """
        pass

    def get_diagnostics(self, velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """項の診断情報を取得

        Args:
            velocity: 現在の速度場
            **kwargs: 追加のパラメータ

        Returns:
            診断情報を含む辞書
        """
        contribution = self.compute(velocity, **kwargs)
        return {
            f"{self.name}_max": max(np.max(np.abs(c)) for c in contribution),
            f"{self.name}_l2norm": np.sqrt(sum(np.sum(c**2) for c in contribution)),
        }

    def __str__(self) -> str:
        """文字列表現"""
        status = "enabled" if self.enabled else "disabled"
        return f"{self.name} term ({status})"
