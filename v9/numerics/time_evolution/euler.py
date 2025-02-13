"""前進オイラー法による時間積分を提供するモジュール"""

from typing import Union

from .base import TimeIntegrator, FieldType
from core.field import ScalarField, VectorField


class ForwardEuler(TimeIntegrator):
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

    def integrate(
        self,
        field: FieldType,
        dt: float,
        derivative: FieldType,
    ) -> FieldType:
        """前進オイラー法で時間積分を実行

        Args:
            field: 現在のフィールド値
            dt: 時間刻み幅
            derivative: フィールドの時間微分（fieldと同じ型）

        Returns:
            更新されたフィールド
        """
        if not isinstance(derivative, type(field)):
            raise TypeError("derivativeはfieldと同じ型である必要があります")

        try:
            if isinstance(field, (ScalarField, VectorField)):
                return self._integrate_field(field, dt, derivative)
            else:
                raise ValueError("Unsupported field type")

        except Exception as e:
            raise RuntimeError(f"Euler積分中にエラー: {e}")

    def _integrate_field(
        self,
        field: Union[ScalarField, VectorField],
        dt: float,
        derivative: Union[ScalarField, VectorField],
    ) -> Union[ScalarField, VectorField]:
        """ScalarFieldまたはVectorFieldの時間積分"""
        # 加算演算子を使用した更新
        new_field = field + dt * derivative

        # 誤差の推定
        error = dt * derivative.norm()
        self._error_history.append(error)

        return new_field

    def compute_timestep(self, field: FieldType, **kwargs) -> float:
        """安定な時間刻み幅を計算"""
        dt = super().compute_timestep(field, **kwargs)
        return self._clip_timestep(dt)

    def get_order(self) -> int:
        """数値スキームの次数を取得"""
        return 1

    def get_error_estimate(self) -> float:
        """誤差の推定値を取得

        前進オイラー法の局所打ち切り誤差は O(dt²)
        """
        if not self._error_history:
            return float("inf")
        return max(self._error_history[-10:])
