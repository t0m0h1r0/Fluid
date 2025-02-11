"""前進オイラー法による時間積分を提供するモジュール

このモジュールは、1次精度の前進オイラー法による時間積分を実装します。
"""

from typing import Callable, TypeVar
import numpy as np
from .base import TimeIntegrator, StateLike

T = TypeVar("T", bound=StateLike)


class ForwardEuler(TimeIntegrator[T]):
    """前進オイラー法による時間積分器

    簡単だが1次精度の明示的時間積分スキーム。
    条件付き安定で、時間刻み幅に制限があります。
    """

    def __init__(
        self,
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        tolerance: float = 1e-6,
    ):
        """前進オイラー法の積分器を初期化

        Args:
            cfl: CFL条件の係数（0 < cfl <= 1）
            min_dt: 最小時間刻み幅
            max_dt: 最大時間刻み幅
            tolerance: 収束判定の許容誤差
        """
        super().__init__(
            cfl=cfl,
            min_dt=min_dt,
            max_dt=max_dt,
            tolerance=tolerance,
            stability_limit=2.0,  # von Neumannの安定性解析による
        )

    def integrate(self, state: T, dt: float, derivative_fn: Callable[[T], T]) -> T:
        """前進オイラー法で時間積分を実行

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数

        Returns:
            更新された状態

        Notes:
            前進オイラー法: u^{n+1} = u^n + dt * f(u^n)
        """
        # 時間微分の計算
        derivative = derivative_fn(state)

        # numpy配列の場合の特別処理
        if isinstance(state, np.ndarray):
            new_state = state + dt * derivative
        else:
            # StateLike オブジェクトの場合
            new_state = state.copy()
            new_state.update(derivative, dt)

        # 誤差の推定
        if hasattr(derivative, "norm"):
            error = dt * derivative.norm()
            self._error_history.append(error)
        elif isinstance(derivative, np.ndarray):
            error = dt * np.linalg.norm(derivative)
            self._error_history.append(error)

        return new_state

    def compute_timestep(self, state: T, **kwargs) -> float:
        """安定な時間刻み幅を計算

        Args:
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅

        Notes:
            前進オイラー法の安定条件: dt <= 2/λ_max
            ここで λ_max は右辺の演算子の最大固有値
        """
        # デフォルトのCFL条件による計算
        dt = super().compute_timestep(state, **kwargs)

        # 安定性限界による制限
        if hasattr(state, "get_max_eigenvalue"):
            lambda_max = state.get_max_eigenvalue()
            if lambda_max > 0:
                dt = min(dt, self._stability_limit / lambda_max)

        return self._clip_timestep(dt)

    def get_order(self) -> int:
        """数値スキームの次数を取得

        Returns:
            スキームの次数 (= 1)
        """
        return 1

    def get_error_estimate(self) -> float:
        """誤差の推定値を取得

        Returns:
            推定された誤差

        Notes:
            前進オイラー法の局所打ち切り誤差は O(dt²)
        """
        if not self._error_history:
            return float("inf")
        return max(self._error_history[-10:]) if self._error_history else float("inf")
