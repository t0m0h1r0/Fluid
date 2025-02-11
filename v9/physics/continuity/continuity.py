"""
連続の方程式（レベルセット関数の移流方程式）を解くためのモジュール

支配方程式: ∂t∂ϕ + u⋅∇ϕ = 0

注意: このモジュールは、LevelSet法における界面追跡のための基本的な移流方程式を実装します。
"""

from typing import Dict, Any
import numpy as np
from core.field import VectorField
from physics.levelset import LevelSetField


class ContinuityEquation:
    """
    レベルセット関数の連続の方程式を解くためのクラス

    この実装は、レベルセット関数の移流を計算し、時間発展を追跡します。
    """

    def __init__(
        self,
        use_weno: bool = True,
        weno_order: int = 5,
        name: str = "LevelSetContinuity",
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
        self, levelset: LevelSetField, velocity: VectorField, dt: float = 0.0
    ) -> np.ndarray:
        """
        レベルセット関数の時間微分を計算

        Args:
            levelset: 現在のレベルセット関数
            velocity: 速度場
            dt: 時間刻み幅（オプション）

        Returns:
            レベルセット関数の時間微分
        """
        # 単純な中心差分による計算
        derivative = np.zeros_like(levelset.data)

        # u⋅∇ϕ の計算
        for i in range(velocity.ndim):
            # 各速度成分と空間微分の積を加算
            derivative -= velocity.components[i].data * levelset.gradient(i)

        return derivative

    def compute_timestep(
        self, velocity: VectorField, levelset: LevelSetField, **kwargs
    ) -> float:
        """
        レベルセット移流のための時間刻み幅を計算

        Args:
            velocity: 速度場
            levelset: レベルセット関数
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅
        """
        # CFLベースの時間刻み幅計算
        cfl = kwargs.get("cfl", 0.5)

        # 最大速度の取得
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # CFL条件に基づく時間刻み幅の計算
        return cfl * levelset.dx / (max_velocity + 1e-10)

    def get_diagnostics(
        self, levelset: LevelSetField, velocity: VectorField
    ) -> Dict[str, Any]:
        """
        診断情報を取得

        Args:
            levelset: レベルセット関数
            velocity: 速度場

        Returns:
            診断情報の辞書
        """
        # 移流項の最大値と特性を計算
        advection_terms = []
        for i in range(velocity.ndim):
            advection_term = velocity.components[i].data * levelset.gradient(i)
            advection_terms.append(advection_term)

        return {
            "name": self.name,
            "max_advection": float(np.max(np.abs(sum(advection_terms)))),
            "method": "WENO" if self.use_weno else "Central",
            "weno_order": self.weno_order if self.use_weno else None,
            "levelset_volume": levelset.volume(),
            "levelset_area": levelset.area(),
        }
