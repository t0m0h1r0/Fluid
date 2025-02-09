"""Navier-Stokes方程式の基底クラスとインターフェースを提供

このモジュールでは、Navier-Stokes方程式の解法に関する
基本的なインターフェースと抽象基底クラスを定義します。
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Tuple, List, Optional
import numpy as np

from core.field import VectorField, ScalarField
import logging


class NavierStokesTerm(Protocol):
    """Navier-Stokes方程式の各項のインターフェース"""

    @property
    def name(self) -> str:
        """項の名前"""
        ...

    def compute(self, velocity: VectorField, dt: float, **kwargs) -> List[np.ndarray]:
        """項の寄与を計算"""
        ...

    def get_diagnostics(self, velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


class TimeIntegrator(Protocol):
    """時間積分スキームのインターフェース"""

    def step(
        self,
        initial_state: VectorField,
        dt: float,
        terms: List[NavierStokesTerm],
        **kwargs,
    ) -> VectorField:
        """1時間ステップを実行"""
        ...


class PressureProjection(Protocol):
    """圧力投影法のインターフェース"""

    def project(
        self, velocity: VectorField, pressure: ScalarField, dt: float, **kwargs
    ) -> Tuple[VectorField, ScalarField]:
        """速度場を非圧縮に投影し、圧力を更新"""
        ...


class NavierStokesBase(ABC):
    """Navier-Stokesソルバーの基底クラス"""

    def __init__(
        self,
        time_integrator: TimeIntegrator,
        pressure_projection: PressureProjection,
        terms: List[NavierStokesTerm],
        logger=None,
    ):
        """初期化

        Args:
            time_integrator: 時間積分スキーム
            pressure_projection: 圧力投影法
            terms: NS方程式の各項
            logger: ロガー
        """
        self.time_integrator = time_integrator
        self.pressure_projection = pressure_projection
        self.terms = terms
        self.logger = logger

        # 診断情報の初期化
        self._iteration_count = 0
        self._total_time = 0.0
        self._diagnostics: Dict[str, Any] = {}

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """ソルバーを初期化"""
        pass

    @abstractmethod
    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """時間刻み幅を計算"""
        pass

    @abstractmethod
    def step_forward(
        self, state, dt: Optional[float] = None, **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """1時間ステップを進める"""
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "iteration_count": self._iteration_count,
            "total_time": self._total_time,
            **self._diagnostics,
        }

    def log_diagnostics(self, level: str = "info"):
        """診断情報をログに出力"""
        if self.logger:
            diag = self.get_diagnostics()
            self.logger.log(getattr(logging, level.upper()), f"Diagnostics: {diag}")
