"""Navier-Stokes方程式のソルバーの基底クラスを提供

このモジュールは、Navier-Stokesソルバーの基本機能を実装する
抽象基底クラスを提供します。
"""

from typing import Dict, Any, List, Optional
from abc import abstractmethod

from core.field import VectorField
from physics.levelset import LevelSetField, LevelSetPropertiesManager
from .interfaces import NavierStokesSolver, NavierStokesTerm, TimeIntegrator


class NavierStokesBase(NavierStokesSolver):
    """Navier-Stokesソルバーの基底クラス"""

    def __init__(
        self,
        time_integrator: TimeIntegrator,
        terms: List[NavierStokesTerm],
        properties: Optional[LevelSetPropertiesManager] = None,
        logger=None,
    ):
        """基底クラスを初期化

        Args:
            time_integrator: 時間積分スキーム
            terms: NS方程式の各項
            properties: 物性値マネージャー
            logger: ロガー
        """
        self.time_integrator = time_integrator
        self.terms = terms
        self.properties = properties
        self.logger = logger

        # 内部状態の初期化
        self._time = 0.0
        self._iteration_count = 0
        self._diagnostics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """ソルバーの名前を取得"""
        return "NavierStokesBase"

    @property
    def time(self) -> float:
        """現在の時刻を取得"""
        return self._time

    @property
    def iteration_count(self) -> int:
        """反復回数を取得"""
        return self._iteration_count

    def initialize(self, **kwargs) -> None:
        """ソルバーを初期化"""
        self._time = 0.0
        self._iteration_count = 0
        self._diagnostics.clear()

    @abstractmethod
    def compute_derivative(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: LevelSetPropertiesManager,
        **kwargs,
    ) -> VectorField:
        """速度場の時間微分を計算"""
        pass

    def compute_timestep(self, **kwargs) -> float:
        """時間刻み幅を計算

        各項の制限のうち最も厳しいものを採用
        """
        state = kwargs.get("state")
        if state is None:
            raise ValueError("stateが指定されていません")

        dt_limits = []
        for term in self.terms:
            dt = term.compute_timestep(
                state.velocity, state.levelset, state.properties, **kwargs
            )
            if dt > 0:  # 有効な制限のみ考慮
                dt_limits.append(dt)

        if not dt_limits:
            raise ValueError("有効な時間刻み幅の制限がありません")

        return min(dt_limits)

    def step_forward(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める"""
        state = kwargs.get("state")
        if state is None:
            raise ValueError("stateが指定されていません")

        try:
            # 時間発展の実行
            next_state = self.time_integrator.step(
                state=state,
                dt=dt,
                compute_derivative=self.compute_derivative,
                **kwargs,
            )

            # 内部状態の更新
            self._time += dt
            self._iteration_count += 1

            # 診断情報の収集
            diagnostics = self._collect_diagnostics(next_state, dt)

            return {
                "state": next_state,
                "diagnostics": diagnostics,
                "time": self._time,
                "dt": dt,
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"時間発展計算中にエラー: {e}")
            raise

    def finalize(self, **kwargs) -> None:
        """ソルバーの終了処理"""
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """ソルバーの診断情報を取得"""
        return {
            "time": self._time,
            "iterations": self._iteration_count,
            "terms": [term.name for term in self.terms],
            **self._diagnostics,
        }

    def _collect_diagnostics(self, state: Any, dt: float) -> Dict[str, Any]:
        """診断情報を収集"""
        diag = {
            "time": self._time,
            "dt": dt,
            "iteration": self._iteration_count,
        }

        # 各項の診断情報を収集
        for term in self.terms:
            diag[term.name] = term.get_diagnostics()

        return diag

    def log(self, level: str, msg: str):
        """ログを出力"""
        if self.logger:
            log_method = getattr(self.logger, level, None)
            if log_method:
                log_method(msg)
