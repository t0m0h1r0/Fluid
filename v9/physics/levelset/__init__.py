"""Level Set法パッケージ

このパッケージは、Level Set法による界面追跡のための機能を提供します。
"""

from dataclasses import dataclass
from typing import Optional

from .base import LevelSetTerm, LevelSetTermBase, LevelSetSolverBase
from .config import LevelSetConfig
from .field import LevelSetField, LevelSetParameters
from .solver import LevelSetSolver
from .utils import (
    heaviside,
    delta,
    compute_curvature,
    compute_volume,
    compute_area,
    extend_velocity,
    interpolate_property,
    compute_property_jump,
    compute_interface_statistics,
)
from .reinitialize import reinitialize_levelset


@dataclass
class FluidProperties:
    """流体の物性値を保持するクラス

    Attributes:
        density: 密度 [kg/m³]
        viscosity: 動粘性係数 [Pa·s]
        surface_tension: 表面張力係数 [N/m]
    """

    density: float
    viscosity: float
    surface_tension: Optional[float] = None

    def __post_init__(self):
        """物性値の妥当性チェック"""
        if self.density <= 0:
            raise ValueError("密度は正の値である必要があります")
        if self.viscosity <= 0:
            raise ValueError("粘性係数は正の値である必要があります")
        if self.surface_tension is not None and self.surface_tension < 0:
            raise ValueError("表面張力係数は非負である必要があります")


__all__ = [
    # 基底クラスとインターフェース
    "LevelSetTerm",
    "LevelSetTermBase",
    "LevelSetSolverBase",
    # 設定クラス
    "LevelSetConfig",
    # フィールドクラス
    "LevelSetField",
    "LevelSetParameters",
    # ソルバー
    "LevelSetSolver",
    # 流体物性値
    "FluidProperties",
    # ユーティリティ関数
    "heaviside",
    "delta",
    "compute_curvature",
    "compute_volume",
    "compute_area",
    "extend_velocity",
    "interpolate_property",
    "compute_property_jump",
    "compute_interface_statistics",
    "reinitialize_levelset",
]
