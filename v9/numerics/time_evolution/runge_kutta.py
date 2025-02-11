"""Runge-Kutta法による時間積分を提供するモジュール

このモジュールは、4次精度のRunge-Kutta法（RK4）による時間積分を実装します。
"""

from typing import Callable, TypeVar, List
import numpy as np
from .base import TimeIntegrator, StateLike

T = TypeVar("T", bound=StateLike)


class RungeKutta4(TimeIntegrator[T]):
    """4次のRunge-Kutta法による時間積分器

    4次精度の明示的時間積分スキーム。
    高精度だが、計算コストが比較的高い特徴があります。
    """

    def __init__(
        self,
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        tolerance: float = 1e-6,
    ):
        """4次Runge-Kutta法の積分器を初期化

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
            stability_limit=2.8,  # von Neumannの安定性解析による
        )
        # RK4の係数
        self._a = [0.0, 0.5, 0.5, 1.0]  # ステージの時刻係数
        self._b = [1 / 6, 1 / 3, 1 / 3, 1 / 6]  # 重み係数

    def integrate(self, state: T, dt: float, derivative_fn: Callable[[T], T]) -> T:
        """4次Runge-Kutta法で時間積分を実行

        Args:
            state: 現在の状態
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数

        Returns:
            更新された状態

        Notes:
            RK4の各ステージ:
            k1 = f(t_n, y_n)
            k2 = f(t_n + dt/2, y_n + dt*k1/2)
            k3 = f(t_n + dt/2, y_n + dt*k2/2)
            k4 = f(t_n + dt, y_n + dt*k3)
            y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
        """
        try:
            # numpy配列の場合の特別処理
            if isinstance(state, np.ndarray):
                k = []
                temp_state = state.copy()

                # 第1ステージ
                k1 = derivative_fn(temp_state)
                k.append(k1)

                # 第2,3ステージ
                for i in range(1, 3):
                    temp_state = state + dt * self._a[i] * k[-1]
                    k.append(derivative_fn(temp_state))

                # 第4ステージ
                temp_state = state + dt * self._a[3] * k[-1]
                k4 = derivative_fn(temp_state)
                k.append(k4)

                # 最終的な状態の更新
                weighted_sum = sum(b * k_i for b, k_i in zip(self._b, k))
                new_state = state + dt * weighted_sum

            else:
                # StateLike オブジェクトの場合
                k = []
                temp_state = state.copy()

                # 第1ステージ
                k1 = derivative_fn(state)
                k.append(k1)

                # 第2,3ステージ
                for i in range(1, 3):
                    temp_state = state.copy()
                    temp_state.update(k[-1], dt * self._a[i])
                    k.append(derivative_fn(temp_state))

                # 第4ステージ
                temp_state = state.copy()
                temp_state.update(k[-1], dt * self._a[3])
                k4 = derivative_fn(temp_state)
                k.append(k4)

                # 最終的な状態の更新
                new_state = state.copy()
                weighted_sum = sum(b * k_i for b, k_i in zip(self._b, k))
                new_state.update(weighted_sum, dt)

            # 誤差の推定
            self._estimate_error(k, dt)

            return new_state

        except Exception as e:
            raise RuntimeError(f"RK4積分中にエラー: {e}")

    def _estimate_error(self, k: List[T], dt: float) -> None:
        """RK4の誤差を推定

        Args:
            k: RK4の各ステージでの微分値
            dt: 時間刻み幅

        Notes:
            エンベディッドRK4(5)法による誤差推定を実装
        """
        if not hasattr(k[0], "norm"):
            return

        # 5次法と4次法の差による誤差推定
        error = dt * abs(k[0].norm() / 6 - k[2].norm() / 3 + k[3].norm() / 6)
        self._error_history.append(error)

    def compute_timestep(self, state: T, **kwargs) -> float:
        """安定な時間刻み幅を計算

        Args:
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅

        Notes:
            RK4の安定条件: dt <= 2.8/λ_max
            ここで λ_max は右辺の演算子の最大固有値
        """
        # デフォルトのCFL条件による計算
        dt = super().compute_timestep(state, **kwargs)

        # 安定性限界による制限
        if hasattr(state, "get_max_eigenvalue"):
            lambda_max = state.get_max_eigenvalue()
            if lambda_max > 0:
                dt = min(dt, self._stability_limit / lambda_max)

        # 誤差履歴に基づく制御
        if self._error_history:
            current_error = self._error_history[-1]
            if current_error > self._tolerance:
                dt *= 0.8  # 時間刻み幅を縮小
            elif current_error < self._tolerance / 10:
                dt *= 1.2  # 時間刻み幅を拡大

        return self._clip_timestep(dt)

    def get_order(self) -> int:
        """数値スキームの次数を取得

        Returns:
            スキームの次数 (= 4)
        """
        return 4

    def get_error_estimate(self) -> float:
        """誤差の推定値を取得

        Returns:
            推定された誤差

        Notes:
            RK4の局所打ち切り誤差は O(dt⁵)
        """
        if not self._error_history:
            return float("inf")
        # 直近の誤差履歴の最大値を返す
        return max(self._error_history[-10:]) if self._error_history else float("inf")
