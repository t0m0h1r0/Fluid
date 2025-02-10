"""時間発展ソルバーの基底クラスを提供するモジュール

このモジュールは、時間発展問題を解くためのソルバーの基底クラスを定義します。
"""

from typing import Dict, Any, Optional, List
import numpy as np

from .base import TimeEvolutionBase, TimeEvolutionTerm, TimeEvolutionConfig
from .integrator import create_integrator


class TimeEvolutionSolver(TimeEvolutionBase):
    """時間発展ソルバーの基底クラス"""

    def __init__(
        self,
        terms: Optional[List[TimeEvolutionTerm]] = None,
        integrator_type: str = "rk4",
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        config: Optional[TimeEvolutionConfig] = None,
        logger=None,
    ):
        """ソルバーを初期化

        Args:
            terms: 時間発展方程式の各項
            integrator_type: 時間積分器の種類
            cfl: CFL数
            min_dt: 最小時間刻み幅
            max_dt: 最大時間刻み幅
            config: 時間発展設定
            logger: ロガー
        """
        super().__init__(config, logger)

        # 時間発展の項
        self.terms: List[TimeEvolutionTerm] = terms or []

        # 時間刻み幅の制限
        self.cfl = cfl
        self.min_dt = min_dt
        self.max_dt = max_dt

        # 時間積分器の設定
        self.integrator = create_integrator(integrator_type)

    def compute_timestep(self, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算

        Args:
            **kwargs: 時間刻み幅計算に必要なパラメータ

        Returns:
            計算された時間刻み幅
        """
        # 各項からの時間刻み幅制限を計算
        dt_limits = [self.max_dt]
        for term in self.terms:
            if hasattr(term, "compute_timestep"):
                try:
                    dt_limits.append(term.compute_timestep(**kwargs))
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"項{term.name}の時間刻み幅計算でエラー: {e}"
                        )

        # CFL条件に基づく制限と項からの制限を統合
        dt = min(dt_limits)

        # 全体の制限を適用
        return np.clip(dt, self.min_dt, self.max_dt)

    def compute_derivative(self, state: Any, **kwargs) -> Any:
        """時間微分を計算

        Args:
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間微分
        """
        # 各項の寄与を計算
        contributions = []
        for term in self.terms:
            try:
                contrib = term.compute(state, **kwargs)
                contributions.append(contrib)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"項{term.name}の計算中にエラー: {e}")
                raise

        return sum(contributions)

    def step_forward(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める

        Args:
            dt: 時間刻み幅
            **kwargs: 追加のパラメータ

        Returns:
            計算結果と診断情報を含む辞書
        """
        try:
            # 状態の取得
            state = kwargs.get("state")
            if state is None:
                raise ValueError("stateが指定されていません")

            # 時間刻み幅の更新
            self._dt = dt

            # 時間積分の実行
            new_state = self.integrator.integrate(
                dt=dt, derivative_fn=self.compute_derivative, state=state, **kwargs
            )

            # 反復回数と時刻の更新
            self._iteration_count += 1
            self._time += dt
            self._time_history.append(self._time)

            # 診断情報の収集
            diagnostics: Dict[str, Any] = {
                "time": self._time,
                "dt": dt,
                "iteration": self._iteration_count,
                "integrator_info": (
                    self.integrator.get_diagnostics()
                    if hasattr(self.integrator, "get_diagnostics")
                    else {}
                ),
            }

            # 各項の診断情報を追加
            for term in self.terms:
                try:
                    diagnostics[term.name] = term.get_diagnostics(
                        state=new_state, previous_state=state, dt=dt
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"項{term.name}の診断情報取得でエラー: {e}")

            return {"state": new_state, "time": self._time, "diagnostics": diagnostics}

        except Exception as e:
            if self.logger:
                self.logger.error(f"時間発展計算中にエラー: {e}")
            raise

    def get_diagnostics(self) -> Dict[str, Any]:
        """ソルバーの診断情報を取得

        Returns:
            診断情報の辞書
        """
        diag = super().get_diagnostics()
        diag.update(
            {
                "active_terms": len(self.terms),
                "integrator": self.integrator.__class__.__name__,
                "cfl": self.cfl,
                "min_dt": self.min_dt,
                "max_dt": self.max_dt,
            }
        )
        return diag
