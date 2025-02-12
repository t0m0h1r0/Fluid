"""
連続の方程式（移流方程式）を解くためのモジュール

支配方程式: ∂t∂f + u⋅∇f = 0

このモジュールは、任意のスカラー場の移流を計算し、時間発展を追跡します。
"""

from typing import Dict, Any
from core.field import VectorField, ScalarField
import numpy as np


class ContinuityEquation:
    """
    連続の方程式（移流方程式）を解くためのクラス

    この実装は、スカラー場の移流を計算し、時間発展を追跡します。
    レベルセット法、温度場、密度場など、様々なスカラー場の移流に使用できます。
    """

    def __init__(
        self,
        use_weno: bool = True,
        weno_order: int = 5,
        name: str = "Continuity",
    ):
        """
        連続の方程式クラスを初期化

        Args:
            use_weno: WENO（Weighted Essentially Non-Oscillatory）スキームを使用するかどうか
            weno_order: WENOスキームの次数
            name: 方程式の名前
        """
        self.use_weno = use_weno
        self.weno_order = weno_order
        self.name = name

    def compute_derivative(
        self, field: ScalarField, velocity: VectorField,
    ) -> ScalarField:
        """
        スカラー場の時間微分を計算

        Args:
            field: 移流される任意のスカラー場
            velocity: 速度場

        Returns:
            スカラー場の時間微分をScalarFieldとして返す
        """
        # 結果を格納するScalarFieldを作成
        result = ScalarField(field.shape, field.dx)

        # 移流項 u⋅∇f の計算
        advection = result.data  # 直接データ配列を参照

        # 各方向の速度成分による寄与を計算
        for i in range(velocity.ndim):
            # 速度場とスカラー場の勾配の積を計算
            # 将来的にはWENOスキームを使用する
            advection += velocity.components[i].data * field.gradient(i)

        # 移流の符号を反転（移流方程式の形式）
        result.data = -advection

        return result

    def compute_timestep(
        self, velocity: VectorField, field: ScalarField, **kwargs
    ) -> float:
        """
        移流のための時間刻み幅を計算

        Args:
            velocity: 速度場
            field: スカラー場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅
        """
        # CFLベースの時間刻み幅計算
        cfl = kwargs.get("cfl", 0.5)

        # 最大速度の取得
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # CFL条件に基づく時間刻み幅の計算
        return cfl * field.dx / (max_velocity + 1e-10)

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
        # 移流項の最大値と特性を計算
        advection_terms = []
        for i in range(velocity.ndim):
            advection_term = velocity.components[i].data * field.gradient(i)
            advection_terms.append(advection_term)

        return {
            "name": self.name,
            "max_advection": float(max(np.max(np.abs(r)) for r in advection_terms)),
            "method": "WENO" if self.use_weno else "Central",
            "weno_order": self.weno_order if self.use_weno else None,
            "field_mean": float(np.mean(field.data)),
            "field_max": float(np.max(field.data)),
            "field_min": float(np.min(field.data)),
        }