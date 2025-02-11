import numpy as np

from .field import LevelSetField
from .reinitializer import reinitialize_levelset
from .utils import extend_velocity
from core.field import VectorField


class LevelSetMethod:
    """Level Set法の主要な計算ロジックを実装"""

    def __init__(self, use_weno: bool = True, weno_order: int = 5):
        """
        Level Set法を初期化

        Args:
            use_weno: WENOスキームを使用するかどうか
            weno_order: WENOスキームの次数
        """
        self.use_weno = use_weno
        self.weno_order = weno_order

    def compute_derivative(
        self, levelset: LevelSetField, velocity: np.ndarray, dt: float = 0.0
    ) -> LevelSetField:
        """Level Set関数の時間微分を計算"""
        if self.use_weno:
            return self._weno_derivative(levelset, velocity, dt)
        else:
            return self._central_derivative(levelset, velocity, dt)

    def _weno_derivative(
        self, levelset: LevelSetField, velocity: np.ndarray, dt: float
    ) -> LevelSetField:
        """WENOスキームによる時間微分の計算"""
        # WENOスキームの実装（複雑な実装は省略）
        raise NotImplementedError("WENO derivative not implemented")

    def _central_derivative(
        self, levelset: LevelSetField, velocity: np.ndarray, dt: float
    ) -> LevelSetField:
        """中心差分による時間微分の計算"""
        # 中心差分の実装
        derivatives = [
            np.gradient(v * comp, levelset.dx, axis=i)
            for i, (v, comp) in enumerate(zip(velocity, levelset.data.T))
        ]

        derivative_data = -sum(derivatives)

        return LevelSetField(
            data=levelset.data + dt * derivative_data,
            dx=levelset.dx,
            params=levelset.params,
        )

    def reinitialize(self, levelset: LevelSetField) -> LevelSetField:
        """Level Set関数を再初期化"""
        return reinitialize_levelset(
            levelset, dt=levelset.params.epsilon, n_steps=levelset.params.reinit_steps
        )

    def extend_velocity(
        self, velocity: np.ndarray, levelset: LevelSetField
    ) -> np.ndarray:
        """界面に沿って速度場を延長"""
        return extend_velocity(velocity, levelset.data, levelset.dx)

    def run(self, levelset: LevelSetField, velocity: VectorField) -> np.ndarray:
        """Level Set関数の時間微分を計算

        Args:
            levelset: Level Set場
            velocity: 速度場（界面の移流に使用）

        Returns:
            Level Set関数の時間微分
        """
        # 連続の式に基づく時間発展方程式の計算
        # -(u⋅∇φ) を計算
        derivative = np.zeros_like(levelset.data)

        for i, u_i in enumerate(velocity.components):
            # 各方向の速度成分と空間微分の積を加算
            derivative -= u_i.data * levelset.gradient(i)

        return derivative
