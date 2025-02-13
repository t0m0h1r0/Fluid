"""
圧力ポアソン方程式の各項の基底クラスを提供するモジュール

このモジュールは、圧力ポアソン方程式を構成する各項（移流、粘性、外力）の
共通インターフェースと基本機能を定義します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from core.field import ScalarField


class PoissonTerm(ABC):
    """圧力ポアソン方程式の項の基底クラス"""

    def __init__(self, name: str = "BaseTerm", enabled: bool = True):
        """
        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        self._name = name
        self._enabled = enabled
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
    def compute(self, **kwargs) -> ScalarField:
        """
        項の寄与を計算する抽象メソッド

        Returns:
            項の寄与を表すScalarField
        """
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {"name": self.name, "enabled": self.enabled, **self._diagnostics}
