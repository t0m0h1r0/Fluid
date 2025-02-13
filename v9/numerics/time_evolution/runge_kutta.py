"""4次のRunge-Kutta法による時間積分を提供するモジュール"""

from typing import Callable, TypeVar, List, Union
from dataclasses import dataclass
from .base import TimeIntegrator, StateLike
from core.field import ScalarField, VectorField

T = TypeVar("T", bound=StateLike)


@dataclass
class RKStage:
    """Runge-Kutta法の各ステージの情報"""

    coefficient: float  # 時刻係数
    weight: float  # 重み係数


class RungeKutta4(TimeIntegrator[T]):
    """4次のRunge-Kutta法による時間積分器

    高精度な明示的時間積分スキーム。
    """

    def __init__(
        self,
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        tolerance: float = 1e-6,
    ):
        """4次Runge-Kutta法の積分器を初期化"""
        super().__init__(
            cfl=cfl,
            min_dt=min_dt,
            max_dt=max_dt,
            tolerance=tolerance,
            stability_limit=2.8,  # von Neumannの安定性解析による
        )
        # RK4の各ステージ設定
        self._stages = [
            RKStage(coefficient=0.0, weight=1 / 6),
            RKStage(coefficient=0.5, weight=1 / 3),
            RKStage(coefficient=0.5, weight=1 / 3),
            RKStage(coefficient=1.0, weight=1 / 6),
        ]

    def integrate(self, state: T, dt: float, derivative_fn: Callable[[T], T]) -> T:
        """4次Runge-Kutta法で時間積分を実行"""
        try:
            if isinstance(state, (ScalarField, VectorField)):
                return self._integrate_field(state, dt, derivative_fn)
            else:
                return self._integrate_state(state, dt, derivative_fn)
        except Exception as e:
            raise RuntimeError(f"RK4積分中にエラー: {e}")

    def _integrate_field(
        self,
        field: Union[ScalarField, VectorField],
        dt: float,
        derivative_fn: Callable[
            [Union[ScalarField, VectorField]], Union[ScalarField, VectorField]
        ],
    ) -> Union[ScalarField, VectorField]:
        """ScalarFieldまたはVectorFieldの時間積分"""
        k_values = []
        temp_field = field.copy()

        # 各ステージの計算
        for stage in self._stages:
            if k_values:  # 第2ステージ以降
                temp_field = field + dt * stage.coefficient * k_values[-1]
            k_values.append(derivative_fn(temp_field))

        # 最終的な更新
        result = field.copy()
        for k, stage in zip(k_values, self._stages):
            result += dt * stage.weight * k

        # 誤差の推定
        self._estimate_error(k_values, dt)
        return result

    def _integrate_state(
        self, state: T, dt: float, derivative_fn: Callable[[T], T]
    ) -> T:
        """一般的な状態オブジェクトの時間積分"""
        k_values = []
        temp_state = state.copy()

        # 各ステージの計算
        for stage in self._stages:
            if k_values:  # 第2ステージ以降
                temp_state = state.copy()
                temp_state.update(k_values[-1], dt * stage.coefficient)
            k_values.append(derivative_fn(temp_state))

        # 最終的な更新
        new_state = state.copy()
        weighted_sum = k_values[0].copy()
        for k, stage in zip(k_values[1:], self._stages[1:]):
            weighted_sum.update(k, stage.weight / self._stages[0].weight)
        new_state.update(weighted_sum, dt)

        # 誤差の推定
        self._estimate_error(k_values, dt)
        return new_state

    def _estimate_error(self, k_values: List[T], dt: float) -> None:
        """RK4の誤差を推定（エンベディッドRK4(5)法による）"""
        if not hasattr(k_values[0], "norm"):
            return

        # 5次法と4次法の差による誤差推定
        error = dt * abs(
            k_values[0].norm() / 6 - k_values[2].norm() / 3 + k_values[3].norm() / 6
        )
        self._error_history.append(error)

    def compute_timestep(self, state: T, **kwargs) -> float:
        """安定な時間刻み幅を計算"""
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
