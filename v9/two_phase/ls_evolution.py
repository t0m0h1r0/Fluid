from typing import Dict, Any, Optional, Type
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.levelset.utils import reinitialize, compute_volume, compute_area
from physics.navier_stokes.timestepping import RungeKutta4, TimeIntegrator

from .base_evolution import BaseEvolution


class LevelSetEvolution(BaseEvolution):
    """Level Set法の進化を管理するクラス"""

    def __init__(
        self,
        epsilon: float = 1.0e-2,
        reinit_interval: int = 5,
        reinit_steps: int = 2,
        max_reinit_iterations: int = 100,
        reinit_tolerance: float = 1.0e-6,
        time_integrator_class: Optional[Type[TimeIntegrator]] = None,
        weno_order: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """Level Set進化クラスを初期化

        Args:
            epsilon: 界面の幅パラメータ
            reinit_interval: 再初期化の間隔
            reinit_steps: 再初期化のステップ数
            max_reinit_iterations: 再初期化の最大反復回数
            reinit_tolerance: 再初期化の収束判定
            time_integrator_class: 時間積分クラス（未指定の場合はRK4）
            weno_order: WENOスキームの次数（Noneの場合は無効）
            *args: BaseEvolutionに渡す位置引数
            **kwargs: BaseEvolutionに渡すキーワード引数
        """
        super().__init__("LevelSet", *args, **kwargs)

        # パラメータの設定
        self.epsilon = epsilon
        self.reinit_interval = reinit_interval
        self.reinit_steps = reinit_steps
        self._steps_since_reinit = 0

        # 再初期化パラメータ
        self.max_reinit_iterations = max_reinit_iterations
        self.reinit_tolerance = reinit_tolerance

        # 時間積分器の設定（デフォルトはRK4）
        self.time_integrator_class = time_integrator_class or RungeKutta4
        self.time_integrator = self.time_integrator_class()

        # WENOスキームの設定
        self.weno_order = weno_order

    def compute_timestep(
        self, velocity: VectorField, level_set: ScalarField, **kwargs
    ) -> float:
        """時間ステップを計算

        Args:
            velocity: 速度場
            level_set: レベルセット場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間ステップ
        """
        # 最大速度を計算
        max_velocity = max(np.max(np.abs(v.data)) for v in velocity.components)

        # CFL条件に基づく時間ステップ
        dt = self.cfl * level_set.dx / (max_velocity + 1e-10)

        return np.clip(dt, self.min_dt, self.max_dt)

    def _compute_flux(
        self, velocity: VectorField, level_set: ScalarField
    ) -> np.ndarray:
        """移流項の計算

        Args:
            velocity: 速度場
            level_set: レベルセット場

        Returns:
            計算された移流フラックス
        """
        # WENOスキームを使用する場合
        if self.weno_order is not None:
            # WENOスキームを使用した高次精度の再構築
            # TODO: WENOスキームの具体的な実装が必要
            raise NotImplementedError(
                "WENOスキームの詳細な実装が必要です。"
                "physics.levelset.utilsなどのモジュールで高次精度の再構築を実装してください。"
            )

        # デフォルトは単純な中心差分
        flux = np.zeros_like(level_set.data)
        for i, v in enumerate(velocity.components):
            # 風上差分
            upwind = v.data < 0
            flux += v.data * np.where(
                upwind,
                np.roll(level_set.data, -1, axis=i),
                np.roll(level_set.data, 1, axis=i),
            )

        return flux

    def advance(
        self,
        current_velocity: VectorField,
        current_level_set: LevelSetField,
        dt: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """レベルセット場を時間発展

        Args:
            current_velocity: 速度場
            current_level_set: レベルセット場
            dt: 時間ステップ
            **kwargs: 追加のパラメータ

        Returns:
            更新されたレベルセット場と診断情報の辞書
        """
        # 移流項の計算
        flux = self._compute_flux(current_velocity, current_level_set)

        # レベルセット場の更新
        new_level_set = current_level_set.copy()
        new_level_set.data = current_level_set.data - dt * flux

        # 再初期化のチェック
        self._steps_since_reinit += 1
        if self._steps_since_reinit >= self.reinit_interval:
            # 再初期化処理
            new_level_set.data = reinitialize(
                new_level_set.data, current_level_set.dx, n_steps=self.reinit_steps
            )
            self._steps_since_reinit = 0

        # 診断情報の生成
        diagnostics = {
            "volume": compute_volume(new_level_set.data, current_level_set.dx),
            "area": compute_area(new_level_set.data, current_level_set.dx),
            "steps_since_reinit": self._steps_since_reinit,
        }

        return {"level_set": new_level_set, "diagnostics": diagnostics}
