"""
連続の方程式（移流方程式）を解くためのモジュール

保存則の形式での移流方程式: ∂f/∂t + ∇⋅(uf) = 0
非保存形式での移流方程式: ∂f/∂t + u⋅∇f = 0

このモジュールは、任意のスカラー場の移流を計算し、時間発展を追跡します。
スカラー場は物質（レベルセット関数）や状態量（温度など）を表現できます。
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField


class ContinuityEquation:
    """
    連続の方程式（移流方程式）を解くためのクラス

    この実装は、スカラー場の移流を計算し、時間発展を追跡します。
    レベルセット関数、温度場、密度場など、様々なスカラー場の移流に使用できます。
    """

    def __init__(self, name: str = "Continuity"):
        """連続の方程式クラスを初期化

        Args:
            name: 方程式の名前（デフォルト: "Continuity"）
        """
        self.name = name

    def compute_derivative(
        self,
        field: ScalarField,
        velocity: VectorField,
    ) -> ScalarField:
        """
        スカラー場の時間微分を計算（非保存形式）

        速度場との内積演算を活用して、∂f/∂t = -u⋅∇f を計算します。

        Args:
            field: 移流される任意のスカラー場
            velocity: 速度場

        Returns:
            スカラー場の時間微分をScalarFieldとして返す
        """
        result = ScalarField(field.shape, field.dx)
        result.data = -(velocity * field.gradient()).data
        return result

    def compute_derivative_conservative(
        self,
        field: ScalarField,
        velocity: VectorField,
    ) -> ScalarField:
        """
        スカラー場の時間微分を計算（保存形式）

        フラックスの発散として、∂f/∂t = -∇⋅(uf) を計算します。

        Args:
            field: 移流される任意のスカラー場
            velocity: 速度場

        Returns:
            スカラー場の時間微分をScalarFieldとして返す
        """
        result = ScalarField(field.shape, field.dx)
        flux = velocity * field
        result.data = -flux.divergence().data
        return result

    def compute_timestep(
        self, velocity: VectorField, field: ScalarField, **kwargs
    ) -> float:
        """
        移流のための時間刻み幅を計算

        CFL条件に基づいて安定な時間刻み幅を計算します。
        dt ≤ CFL * dx / max(|u|)

        Args:
            velocity: 速度場
            field: スカラー場
            **kwargs: 追加のパラメータ（cfl係数など）

        Returns:
            計算された時間刻み幅
        """
        # CFLベースの時間刻み幅計算
        cfl = kwargs.get("cfl", 0.5)

        # 各方向の最大速度を計算
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # CFL条件に基づく時間刻み幅の計算
        if max_velocity > 0:
            dt = cfl * field.dx / max_velocity
        else:
            dt = float("inf")

        return dt

    def get_diagnostics(
        self, field: ScalarField, velocity: VectorField
    ) -> Dict[str, Any]:
        """
        診断情報を取得

        Args:
            field: スカラー場
            velocity: 速度場

        Returns:
            診断情報の辞書
        """
        # 移流速度の特性を計算
        velocity_info = {
            f"max_velocity_{dim}": float(np.max(np.abs(comp.data)))
            for dim, comp in zip(["x", "y", "z"], velocity.components)
        }

        # スカラー場の統計情報
        field_info = {
            "mean": float(np.mean(field.data)),
            "max": float(np.max(field.data)),
            "min": float(np.min(field.data)),
            "std": float(np.std(field.data)),
        }

        return {
            "name": self.name,
            "scheme": "central_difference",
            "velocities": velocity_info,
            "field": field_info,
            "grid_spacing": field.dx,
        }
