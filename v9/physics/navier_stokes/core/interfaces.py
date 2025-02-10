"""Navier-Stokes方程式に関する基本インターフェースを定義

このモジュールは、Navier-Stokes方程式のソルバー、項、状態などの
基本的なインターフェースを定義します。
"""

from abc import abstractmethod
from typing import Protocol, Dict, Any, List, Optional
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField, LevelSetPropertiesManager


class NSComponent(Protocol):
    """Navier-Stokes方程式の構成要素の基本インターフェース"""

    @property
    def name(self) -> str:
        """コンポーネントの名前"""
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


class NavierStokesTerm(NSComponent, Protocol):
    """Navier-Stokes方程式の項のインターフェース"""

    @abstractmethod
    def compute(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: LevelSetPropertiesManager,
        **kwargs,
    ) -> List[np.ndarray]:
        """項の寄与を計算"""
        ...

    @abstractmethod
    def compute_timestep(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: LevelSetPropertiesManager,
        **kwargs,
    ) -> float:
        """項に基づく時間刻み幅の制限を計算"""
        ...


class TimeIntegrator(Protocol):
    """時間積分スキームのインターフェース"""

    @abstractmethod
    def step(
        self,
        state: Any,
        dt: float,
        compute_derivative: Any,
        **kwargs,
    ) -> Any:
        """1時間ステップを実行"""
        ...


class PressureProjection(Protocol):
    """圧力投影法のインターフェース"""

    @abstractmethod
    def project(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        dt: float,
        levelset: Optional[LevelSetField] = None,
        properties: Optional[LevelSetPropertiesManager] = None,
    ) -> tuple[VectorField, ScalarField]:
        """速度場を非圧縮に投影し、圧力を更新"""
        ...


class NavierStokesSolver(NSComponent, Protocol):
    """Navier-Stokesソルバーのインターフェース"""

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """ソルバーを初期化"""
        ...

    @abstractmethod
    def step_forward(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める"""
        ...

    @abstractmethod
    def compute_timestep(self, **kwargs) -> float:
        """時間刻み幅を計算"""
        ...

    @abstractmethod
    def finalize(self, **kwargs) -> None:
        """ソルバーの終了処理"""
        ...
