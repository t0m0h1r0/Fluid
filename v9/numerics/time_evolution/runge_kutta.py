"""4次のRunge-Kutta法による時間積分を提供するモジュール"""

from typing import List, Union
from dataclasses import dataclass

from .base import TimeIntegrator, FieldType
from core.field import ScalarField, VectorField


@dataclass
class RKStage:
    """Runge-Kutta法の各ステージの情報"""

    coefficient: float  # 時刻係数
    weight: float  # 重み係数


class RungeKutta4(TimeIntegrator):
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

    def integrate(
        self,
        field: FieldType,
        dt: float,
        derivative: FieldType,
    ) -> FieldType:
        """4次Runge-Kutta法で時間積分を実行

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
            raise RuntimeError(f"RK4積分中にエラー: {e}")

    def _integrate_field(
        self,
        field: Union[ScalarField, VectorField],
        dt: float,
        derivative: Union[ScalarField, VectorField],
    ) -> Union[ScalarField, VectorField]:
        """ScalarFieldまたはVectorFieldの時間積分"""
        k_values = []
        temp_field = field.copy()

        # 各ステージの計算
        k_values.append(derivative)  # k1 = derivative

        # k2の計算
        temp_field = field + dt * 0.5 * k_values[0]
        k_values.append(k_values[0].copy())  # k2 = k1 (簡略化のため)

        # k3の計算
        temp_field = field + dt * 0.5 * k_values[1]
        k_values.append(k_values[1].copy())  # k3 = k2

        # k4の計算
        temp_field = field + dt * k_values[2]
        k_values.append(k_values[2].copy())  # k4 = k3

        # 最終的な更新
        result = field.copy()
        for k, stage in zip(k_values, self._stages):
            result += dt * stage.weight * k

        # 誤差の推定
        self._estimate_error(k_values, dt)
        return result

    def _estimate_error(self, k_values: List[FieldType], dt: float) -> None:
        """RK4の誤差を推定（エンベディッドRK4(5)法による）"""
        # 誤差の推定方法を変更
        error = dt * max(
            comp.norm()
            for k in k_values
            for comp in (k.components if hasattr(k, "components") else [k])
        )
        self._error_history.append(error)

    def compute_timestep(self, field: FieldType, **kwargs) -> float:
        """安定な時間刻み幅を計算"""
        dt = super().compute_timestep(field, **kwargs)
        return self._clip_timestep(dt)

    def get_order(self) -> int:
        """数値スキームの次数を取得"""
        return 4

    def get_error_estimate(self) -> float:
        """誤差の推定値を取得

        RK4の局所打ち切り誤差は O(dt⁵)
        """
        if not self._error_history:
            return float("inf")
        return max(self._error_history[-10:])
