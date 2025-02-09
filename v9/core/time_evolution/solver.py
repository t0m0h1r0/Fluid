"""時間発展ソルバーの基底クラスを提供するモジュール

このモジュールは、時間発展方程式を解くためのソルバーの基底クラスを定義します。
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
from logging import Logger
from .base import TimeEvolutionBase
from .integrator import create_integrator


class TimeEvolutionSolver(TimeEvolutionBase):
    """時間発展ソルバーの基底クラス"""

    def __init__(
        self,
        integrator_type: str = "rk4",
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        logger: Optional[Logger] = None,
    ):
        """初期化

        Args:
            integrator_type: 時間積分器の種類
            cfl: CFL数
            min_dt: 最小時間刻み幅
            max_dt: 最大時間刻み幅
            logger: ロガー
        """
        super().__init__(logger)
        self.integrator = create_integrator(integrator_type)
        self.cfl = cfl
        self.min_dt = min_dt
        self.max_dt = max_dt
        self._iteration_count = 0

    @abstractmethod
    def compute_derivative(self, state: Any, **kwargs) -> Any:
        """時間微分を計算

        Args:
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間微分
        """
        pass

    def step_forward(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める

        Args:
            dt: 時間刻み幅
            **kwargs: 追加のパラメータ

        Returns:
            計算された結果と診断情報
        """
        # 時間刻み幅の調整
        dt = min(max(dt, self.min_dt), self.max_dt)
        self._dt = dt

        try:
            # 現在の状態を取得
            state = kwargs.pop("state", None)
            if state is None:
                raise ValueError("stateが指定されていません")

            # 時間積分を実行
            new_state = self.integrator.integrate(
                state=state, dt=dt, derivative_fn=self.compute_derivative, **kwargs
            )

            # 反復回数と時刻の更新
            self._iteration_count += 1
            self._time += dt

            # 結果と診断情報を返す
            return {
                "state": new_state,
                "time": self._time,
                "dt": dt,
                "iteration": self._iteration_count,
                "integrator_info": self.integrator.get_diagnostics(),
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"時間発展計算中にエラー: {e}")
            raise

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "iteration_count": self._iteration_count,
                "cfl": self.cfl,
                "integrator_type": self.integrator.__class__.__name__,
            }
        )
        return diag

    def initialize(self, **kwargs) -> None:
        """初期化処理"""
        super().initialize(**kwargs)
        self._iteration_count = 0
