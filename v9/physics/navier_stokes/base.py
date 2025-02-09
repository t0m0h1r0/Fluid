"""Navier-Stokes方程式の基底クラスとインターフェースを提供"""

from abc import ABC
from typing import Protocol, Dict, Any, Tuple, List, Optional
import numpy as np
import logging
from core.field import VectorField, ScalarField


class NSComponentBase:
    """Navier-Stokes関連コンポーネントの基底クラス"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """初期化

        Args:
            logger: ロガー
        """
        self._logger = logger

    def log(self, level: str, msg: str):
        """ログを出力"""
        if self._logger:
            self._logger.log(getattr(logging, level.upper()), msg)


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


class NavierStokesBase(NSComponentBase, ABC):
    """Navier-Stokesソルバーの基底クラス"""

    def __init__(
        self,
        time_integrator: TimeIntegrator,
        pressure_projection: PressureProjection,
        terms: List[NavierStokesTerm],
        logger: Optional[logging.Logger] = None,
    ):
        """初期化"""
        super().__init__(logger)
        self.time_integrator = time_integrator
        self.pressure_projection = pressure_projection
        self.terms = terms

        # 診断情報の初期化
        self._iteration_count = 0
        self._total_time = 0.0
        self._diagnostics: Dict[str, Any] = {}
