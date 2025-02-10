"""Navier-Stokes方程式の各項の基底クラスを提供

このモジュールは、Navier-Stokes方程式の各項（移流項、粘性項など）の
基底となる抽象クラスを提供します。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

from core.field import VectorField
from physics.levelset import LevelSetField
from physics.levelset.properties import PropertiesManager
from ..core.interfaces import NavierStokesTerm


class TermBase(NavierStokesTerm, ABC):
    """Navier-Stokes方程式の項の基底クラス"""

    def __init__(self, name: str, enabled: bool = True, logger=None):
        """基底クラスを初期化

        Args:
            name: 項の名前
            enabled: 項を有効にするかどうか
            logger: ロガー
        """
        self._name = name
        self._enabled = enabled
        self.logger = logger

        # 診断情報の初期化
        self._diagnostics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """項の名前を取得"""
        return self._name

    @property
    def enabled(self) -> bool:
        """項が有効かどうかを取得"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """項の有効/無効を設定"""
        self._enabled = value

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

    def compute_timestep(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> float:
        """項に基づく時間刻み幅の制限を計算

        Args:
            velocity: 速度場
            levelset: レベルセット場
            properties: 物性値マネージャー
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅の制限（制限なしの場合はfloat('inf')）
        """
        return float("inf")

    def get_diagnostics(self) -> Dict[str, Any]:
        """項の診断情報を取得"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self._diagnostics,
        }

    def log(self, level: str, msg: str):
        """ログを出力"""
        if self.logger:
            log_method = getattr(self.logger, level, None)
            if log_method:
                log_method(msg)

    def _update_diagnostics(self, key: str, value: Any):
        """診断情報を更新"""
        self._diagnostics[key] = value


class ViscousTerm(TermBase, ABC):
    """粘性に関連する項の基底クラス"""

    def compute_timestep(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> float:
        """粘性による時間刻み幅の制限を計算

        dx² / (2ν) の形式の制限を実装
        """
        if not self.enabled:
            return float("inf")

        # 動粘性係数の最大値を取得
        nu = properties.get_kinematic_viscosity(levelset)
        nu_max = np.max(nu)

        # 安定性条件に基づく時間刻み幅の制限
        dx = velocity.dx
        return 0.5 * dx**2 / (nu_max + 1e-10)  # ゼロ除算防止


class AdvectiveTerm(TermBase, ABC):
    """移流に関連する項の基底クラス"""

    def compute_timestep(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> float:
        """移流による時間刻み幅の制限を計算

        CFL条件に基づく制限を実装
        """
        if not self.enabled:
            return float("inf")

        # 速度の最大値を計算
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # CFL条件に基づく時間刻み幅の制限
        cfl = kwargs.get("cfl", 0.5)  # デフォルトのCFL数
        dx = velocity.dx
        return cfl * dx / (max_velocity + 1e-10)  # ゼロ除算防止
