"""
Navier-Stokes方程式の項の基底クラスを提供するモジュール

各項（移流、拡散、圧力、加速度など）の共通インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from core.field import VectorField


class BaseNavierStokesTerm(ABC):
    """
    Navier-Stokes方程式の各項の基底抽象クラス

    すべての項に共通の基本機能と、必要なインターフェースを提供します。
    """

    def __init__(self, name: str = "BaseTerm", enabled: bool = True):
        """
        基底クラスの初期化

        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        self._name = name
        self._enabled = enabled
        # 診断情報用の辞書
        self._diagnostics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """項の名前を取得"""
        return self._name

    @property
    def enabled(self) -> bool:
        """項が有効かどうかを取得"""
        return self._enabled

    @abstractmethod
    def compute(self, velocity: VectorField, **kwargs) -> VectorField:
        """
        項の寄与を計算する抽象メソッド

        Args:
            velocity: 速度場
            **kwargs: 追加のパラメータ

        Returns:
            計算された項の寄与をVectorFieldとして返す
        """
        pass

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """
        項に基づく時間刻み幅の制限を計算するデフォルトメソッド

        Args:
            velocity: 速度場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅の制限
        """
        return float("inf")

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        診断情報を取得

        Returns:
            項の診断情報を含む辞書
        """
        base_diagnostics = {
            "name": self.name,
            "enabled": self.enabled,
        }
        base_diagnostics.update(self._diagnostics)
        return base_diagnostics
