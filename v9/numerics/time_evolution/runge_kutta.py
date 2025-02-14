"""4次のRunge-Kutta法による時間積分を提供するモジュール（改良版）"""

from typing import List, Union
from dataclasses import dataclass

from .base import TimeIntegrator, FieldType
from core.field import ScalarField, VectorField


@dataclass
class RKStage:
    """Runge-Kutta法の各ステージの情報"""

    coefficient: float
    weight: float


class RungeKutta4(TimeIntegrator):
    """4次のRunge-Kutta法による時間積分器（改良版）"""

    def __init__(
        self,
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        tolerance: float = 1e-6,
    ):
        super().__init__(
            cfl=cfl,
            min_dt=min_dt,
            max_dt=max_dt,
            tolerance=tolerance,
            stability_limit=2.8,
        )
        self._stages = [
            RKStage(coefficient=0.0, weight=1 / 6),
            RKStage(coefficient=0.5, weight=1 / 3),
            RKStage(coefficient=0.5, weight=1 / 3),
            RKStage(coefficient=1.0, weight=1 / 6),
        ]

    def integrate(
        self,
        field: FieldType,
        dt: float,
        derivative: FieldType,
    ) -> FieldType:
        """4次Runge-Kutta法で時間積分を実行（新しい演算子を活用）"""
        if not isinstance(derivative, type(field)):
            raise TypeError("derivativeはfieldと同じ型である必要があります")

        try:
            if isinstance(field, (ScalarField, VectorField)):
                return self._integrate_field(field, dt, derivative)
            else:
                raise ValueError("Unsupported field type")

        except Exception as e:
            raise RuntimeError(f"RK4積分中にエラー: {e}")

    def _integrate_field(
        self,
        field: Union[ScalarField, VectorField],
        dt: float,
        derivative: Union[ScalarField, VectorField],
    ) -> Union[ScalarField, VectorField]:
        """ScalarFieldまたはVectorFieldの時間積分（新しい演算子を活用）"""
        k_values = []
        temp_field = field.copy()

        # 各ステージの計算（新しい演算子を活用）
        k_values.append(derivative)  # k1 = derivative

        # k2の計算（新しい * と + 演算子を活用）
        temp_field = field + (dt * 0.5) * k_values[0]
        k_values.append(k_values[0].copy())

        # k3の計算
        temp_field = field + (dt * 0.5) * k_values[1]
        k_values.append(k_values[1].copy())

        # k4の計算
        temp_field = field + dt * k_values[2]
        k_values.append(k_values[2].copy())

        # 最終的な更新（新しい演算子を活用）
        result = field.copy()
        for k, stage in zip(k_values, self._stages):
            result += dt * stage.weight * k

        # 誤差の推定
        self._estimate_error(k_values, dt)
        return result

    def _estimate_error(self, k_values: List[FieldType], dt: float) -> None:
        """RK4の誤差を推定（新しい演算子を活用）"""
        error = dt * max(
            comp.norm()
            for k in k_values
            for comp in (k.components if hasattr(k, "components") else [k])
        )
        self._error_history.append(error)
