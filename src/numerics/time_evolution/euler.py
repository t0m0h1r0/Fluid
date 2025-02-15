"""
前進オイラー法による時間積分

1次精度の明示的時間積分スキーム
"""

from typing import Union, Dict, Any

from .base import TimeIntegrator, TimeIntegratorConfig
from core.field import ScalarField, VectorField


class ForwardEuler(TimeIntegrator):
    """
    前進オイラー法による時間積分器

    特徴:
    - 1次精度の単純な時間積分スキーム
    - 計算が軽く、安定性に制限がある
    """

    def __init__(
        self,
        config: TimeIntegratorConfig = TimeIntegratorConfig(
            cfl=0.5,
            stability_limit=1.0,  # von Neumannの安定性解析による
        ),
    ):
        """
        前進オイラー法の積分器を初期化

        Args:
            config: 時間積分の設定パラメータ
        """
        super().__init__(config)

    def integrate(
        self,
        field: Union[ScalarField, VectorField],
        dt: float,
        derivative: Union[ScalarField, VectorField],
    ) -> Union[ScalarField, VectorField]:
        """
        前進オイラー法で時間積分を実行

        数値スキーム: u(t+Δt) = u(t) + Δt * ∂u/∂t

        Args:
            field: 現在の場
            dt: 時間刻み幅
            derivative: 場の時間微分

        Returns:
            更新された場
        """
        # 型と整合性のチェック
        if type(field) != type(derivative):
            raise TypeError("fieldとderivativeは同じ型である必要があります")

        try:
            # 時間発展の計算
            new_field = field + dt * derivative

            # 誤差推定
            error = dt * derivative.norm()
            self._error_history.append(error)

            return new_field

        except Exception as e:
            raise RuntimeError(f"Euler積分中にエラー: {e}")

    def compute_timestep(
        self, field: Union[ScalarField, VectorField], **kwargs
    ) -> float:
        """
        CFL条件に基づく安定な時間刻み幅を計算

        数値的安定性条件:
        Δt ≤ CFL * Δx / |u|_max

        Args:
            field: 現在の場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅
        """
        # 速度の最大大きさを推定（ベクトル場の場合）
        if isinstance(field, VectorField):
            max_velocity = field.magnitude().max()
        else:
            # スカラー場の場合は意味のある値を計算できないため、最大値を使用
            max_velocity = field.max()

        # CFL条件に基づく時間刻み幅の計算
        min_dx = min(self.config.dx) if hasattr(self.config, "dx") else 1.0

        # 0除算を防ぐ
        if max_velocity > 0:
            return self.config.cfl * min_dx / max_velocity
        else:
            return self.config.max_dt

    def get_stability_diagnostics(self) -> Dict[str, Any]:
        """
        安定性に関する診断情報を取得

        Returns:
            安定性診断情報の辞書
        """
        return {
            "method": "Forward Euler",
            "order": 1,  # 1次精度
            "error_history": self._error_history,
            "max_error": max(self._error_history) if self._error_history else None,
            "stability_properties": {
                "conditionally_stable": True,
                "requires_small_timestep": True,
                "first_order_accuracy": True,
            },
        }
