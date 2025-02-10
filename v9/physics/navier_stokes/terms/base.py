from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

from core.field import VectorField


class BaseNavierStokesTerm(ABC):
    """Navier-Stokes方程式の項の基底クラス"""

    def __init__(self, name: str = "BaseTerm", enabled: bool = True):
        self._name = name
        self._enabled = enabled
        self._diagnostics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @abstractmethod
    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        """項の寄与を計算する抽象メソッド"""
        pass

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """デフォルトの時間刻み幅計算"""
        return float("inf")

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {"name": self.name, "enabled": self.enabled, **self._diagnostics}
