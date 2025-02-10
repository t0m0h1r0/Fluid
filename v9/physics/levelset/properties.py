"""Level Set法における物性値と界面特性の管理モジュール

このモジュールは、二相流体シミュレーションにおける物性値の計算と
界面特性の追跡を統合的に管理します。
"""

from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np

from .utils import heaviside


class PhaseProperties(Protocol):
    """相の物性値プロトコル"""

    @property
    def density(self) -> float:
        """密度"""
        ...

    @property
    def viscosity(self) -> float:
        """粘性係数"""
        ...

    @property
    def surface_tension(self) -> Optional[float]:
        """表面張力係数"""
        ...


@dataclass
class FluidPhaseProperties:
    """具体的な流体相の物性値"""

    density: float
    viscosity: float
    surface_tension: Optional[float] = None

    def __post_init__(self):
        """物性値の妥当性検証"""
        if self.density <= 0:
            raise ValueError("密度は正の値である必要があります")
        if self.viscosity <= 0:
            raise ValueError("粘性係数は正の値である必要があります")
        if self.surface_tension is not None and self.surface_tension < 0:
            raise ValueError("表面張力係数は非負である必要があります")


class LevelSetPropertiesManager:
    """Level Set法における物性値管理クラス"""

    def __init__(
        self, phase1: PhaseProperties, phase2: PhaseProperties, epsilon: float = 1.0e-2
    ):
        """物性値マネージャーを初期化

        Args:
            phase1: 第1相の物性値
            phase2: 第2相の物性値
            epsilon: 界面厚さ
        """
        self._phase1 = phase1
        self._phase2 = phase2
        self._epsilon = epsilon

    def compute_density(self, levelset: np.ndarray) -> np.ndarray:
        """密度場を計算

        Args:
            levelset: Level Set場

        Returns:
            密度場
        """
        # Heaviside関数による補間
        H = heaviside(levelset, self._epsilon)
        return self._phase1.density * H + self._phase2.density * (1 - H)

    def compute_viscosity(
        self, levelset: np.ndarray, method: str = "arithmetic"
    ) -> np.ndarray:
        """粘性係数場を計算

        Args:
            levelset: Level Set場
            method: 補間方法（'arithmetic' または 'harmonic'）

        Returns:
            粘性係数場
        """
        # Heaviside関数による補間
        H = heaviside(levelset, self._epsilon)

        # 補間方法の選択
        if method == "harmonic":
            return 1.0 / (H / self._phase1.viscosity + (1 - H) / self._phase2.viscosity)
        else:  # デフォルトは算術平均
            return self._phase1.viscosity * H + self._phase2.viscosity * (1 - H)

    def compute_surface_tension(self, levelset: np.ndarray) -> Optional[np.ndarray]:
        """表面張力係数場を計算

        Args:
            levelset: Level Set場

        Returns:
            表面張力係数場（表面張力が定義されていない場合はNone）
        """
        # 両相の表面張力係数が定義されている場合のみ計算
        if self._phase1.surface_tension is None or self._phase2.surface_tension is None:
            return None

        # 平均表面張力係数
        surface_tension_coeff = 0.5 * (
            self._phase1.surface_tension + self._phase2.surface_tension
        )

        return surface_tension_coeff
