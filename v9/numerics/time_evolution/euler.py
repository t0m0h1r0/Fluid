"""前進オイラー法による時間積分を提供するモジュール"""

from typing import Callable, TypeVar, Union
from .base import TimeIntegrator, StateLike
from core.field import ScalarField, VectorField

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
        """前進オイラー法の積分器を初期化"""
        super().__init__(
            cfl=cfl,
            min_dt=min_dt,
            max_dt=max_dt,
            tolerance=tolerance,
            stability_limit=2.0,  # von Neumannの安定性解析による
        )

    def integrate(self, state: T, dt: float, derivative_fn: Callable[[T], T]) -> T:
        """前進オイラー法で時間積分を実行"""
        try:
            # Field型の場合の特別処理
            if isinstance(state, (ScalarField, VectorField)):
                return self._integrate_field(state, dt, derivative_fn)
            else:
                return self._integrate_state(state, dt, derivative_fn)
        except Exception as e:
            raise RuntimeError(f"Euler積分中にエラー: {e}")

    def _integrate_field(
        self,
        field: Union[ScalarField, VectorField],
        dt: float,
        derivative_fn: Callable[
            [Union[ScalarField, VectorField]], Union[ScalarField, VectorField]
        ],
    ) -> Union[ScalarField, VectorField]:
        """ScalarFieldまたはVectorFieldの時間積分"""
        # 時間微分の計算
        derivative = derivative_fn(field)

        # 演算子を使用した更新
        new_field = field + dt * derivative

        # 誤差の推定
        if isinstance(derivative, (ScalarField, VectorField)):
            error = dt * derivative.norm()
            self._error_history.append(error)

        return new_field

    def _integrate_state(
        self, state: T, dt: float, derivative_fn: Callable[[T], T]
    ) -> T:
        """一般的な状態オブジェクトの時間積分"""
        # 時間微分の計算
        derivative = derivative_fn(state)

        # 状態の更新
        new_state = state.copy()
        new_state.update(derivative, dt)

        # 誤差の推定
        if hasattr(derivative, "norm"):
            error = dt * derivative.norm()
            self._error_history.append(error)

        return new_state

    def compute_timestep(self, state: T, **kwargs) -> float:
        """安定な時間刻み幅を計算"""
        # デフォルトのCFL条件による計算
        dt = super().compute_timestep(state, **kwargs)

        # 安定性限界による制限
        if hasattr(state, "get_max_eigenvalue"):
            lambda_max = state.get_max_eigenvalue()
            if lambda_max > 0:
                dt = min(dt, self._stability_limit / lambda_max)

        # オプションのパラメータ処理
        if "characteristic_speed" in kwargs:
            speed = kwargs["characteristic_speed"]
            dt = min(dt, self.cfl * state.dx / speed)

        return self._clip_timestep(dt)
