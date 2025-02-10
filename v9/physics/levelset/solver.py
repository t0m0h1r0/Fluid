"""Level Set法のソルバーを提供するモジュール

このモジュールは、Level Set方程式の時間発展を計算するためのソルバーを提供します。
"""

from typing import Dict, Any, Optional, List
import numpy as np

from physics.time_evolution import TimeEvolutionSolver
from .base import LevelSetSolverBase, LevelSetTerm
from .field import LevelSetField
from .reinitialize import reinitialize_levelset


class LevelSetSolver(LevelSetSolverBase, TimeEvolutionSolver):
    """Level Set法のソルバークラス"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        terms: Optional[List[LevelSetTerm]] = None,
        logger=None,
    ):
        """ソルバーを初期化

        Args:
            config: ソルバー設定
            terms: Level Set方程式の各項
            logger: ロガー
        """
        # デフォルト設定
        config = config or {}

        # 基底クラスの初期化
        LevelSetSolverBase.__init__(
            self,
            use_weno=config.get("discretization", {}).get("scheme", "weno") == "weno",
            weno_order=config.get("discretization", {}).get("order", 5),
            logger=logger,
        )
        TimeEvolutionSolver.__init__(
            self, logger=logger, cfl=0.5, min_dt=1e-6, max_dt=1.0
        )

        # 項の設定
        self.terms = terms or []

    def compute_derivative(self, state: Any, **kwargs) -> LevelSetField:
        """Level Set関数の時間微分を計算

        Args:
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間微分
        """
        velocity = kwargs.get("velocity")
        if velocity is None:
            raise ValueError("速度場が指定されていません")

        result = LevelSetField(
            state.levelset.shape,
            state.levelset.dx,
            state.levelset.config,
            state.levelset.params,
        )

        if self.use_weno:
            # WENOスキームによる空間離散化
            flux = np.zeros_like(state.levelset.data)
            for i, v in enumerate(velocity.components):
                # 風上差分の方向を決定
                upwind = v.data < 0

                # 正の速度に対する flux
                phi_plus = self._compute_weno_reconstruction(state.levelset.data, i)
                # 負の速度に対する flux
                phi_minus = self._compute_weno_reconstruction(
                    np.flip(state.levelset.data, i), i
                )
                phi_minus = np.flip(phi_minus, i)

                # 風上方向に応じてfluxを選択
                flux += v.data * np.where(upwind, phi_minus, phi_plus)

            result.data = -flux

        else:
            # 標準的な中心差分
            flux = sum(
                v.data * state.levelset.gradient(i)
                for i, v in enumerate(velocity.components)
            )
            result.data = -flux

        return result

    def _compute_weno_reconstruction(self, values: np.ndarray, axis: int) -> np.ndarray:
        """WENOスキームによる再構築の具体的な実装

        Args:
            values: 再構築する値の配列
            axis: 再構築を行う軸

        Returns:
            再構築された値
        """
        # WENO5の実装
        if self.weno_order == 5:
            v1 = np.roll(values, 2, axis=axis)
            v2 = np.roll(values, 1, axis=axis)
            v3 = values
            v4 = np.roll(values, -1, axis=axis)
            v5 = np.roll(values, -2, axis=axis)

            # 各ステンシルでの滑らかさ指標を計算
            eps = 1e-6  # ゼロ除算防止用
            beta0 = (
                13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 1 / 4 * (v1 - 4 * v2 + 3 * v3) ** 2
            )
            beta1 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 1 / 4 * (v2 - v4) ** 2
            beta2 = (
                13 / 12 * (v3 - 2 * v4 + v5) ** 2 + 1 / 4 * (3 * v3 - 4 * v4 + v5) ** 2
            )

            # 非線形重みを計算
            weights = np.array([0.1, 0.6, 0.3])
            alpha0 = weights[0] / (eps + beta0) ** 2
            alpha1 = weights[1] / (eps + beta1) ** 2
            alpha2 = weights[2] / (eps + beta2) ** 2
            alpha_sum = alpha0 + alpha1 + alpha2

            omega0 = alpha0 / alpha_sum
            omega1 = alpha1 / alpha_sum
            omega2 = alpha2 / alpha_sum

            # 各ステンシルでの補間値を計算
            weno_coeffs = [
                [1 / 3, -7 / 6, 11 / 6],  # 左側ステンシル
                [-1 / 6, 5 / 6, 1 / 3],  # 中央ステンシル
                [1 / 3, 5 / 6, -1 / 6],  # 右側ステンシル
            ]

            p0 = (
                weno_coeffs[0][0] * v1 + weno_coeffs[0][1] * v2 + weno_coeffs[0][2] * v3
            )

            p1 = (
                weno_coeffs[1][0] * v2 + weno_coeffs[1][1] * v3 + weno_coeffs[1][2] * v4
            )

            p2 = (
                weno_coeffs[2][0] * v3 + weno_coeffs[2][1] * v4 + weno_coeffs[2][2] * v5
            )

            return omega0 * p0 + omega1 * p1 + omega2 * p2

        # WENO3の実装
        elif self.weno_order == 3:
            v1 = np.roll(values, 1, axis=axis)
            v2 = values
            v3 = np.roll(values, -1, axis=axis)

            # 各ステンシルでの滑らかさ指標を計算
            beta0 = (v2 - v1) ** 2
            beta1 = (v3 - v2) ** 2

            eps = 1e-6
            weights = np.array([1 / 3, 2 / 3])
            alpha0 = weights[0] / (eps + beta0) ** 2
            alpha1 = weights[1] / (eps + beta1) ** 2
            alpha_sum = alpha0 + alpha1

            omega0 = alpha0 / alpha_sum
            omega1 = alpha1 / alpha_sum

            weno_coeffs = [
                [-1 / 2, 3 / 2],  # 左側ステンシル
                [1 / 2, 1 / 2],  # 右側ステンシル
            ]

            p0 = weno_coeffs[0][0] * v1 + weno_coeffs[0][1] * v2
            p1 = weno_coeffs[1][0] * v2 + weno_coeffs[1][1] * v3

            return omega0 * p0 + omega1 * p1

        else:
            raise ValueError(f"未対応のWENO次数です: {self.weno_order}")

    def compute_timestep(self, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算

        Args:
            **kwargs: 必要なパラメータ
                - state: 現在のシミュレーション状態

        Returns:
            計算された時間刻み幅
        """
        state = kwargs.get("state")
        if state is None:
            raise ValueError("stateが指定されていません")

        # 速度の最大値を計算
        velocity = state.velocity
        max_velocity = max(np.max(np.abs(comp.data)) for comp in velocity.components)

        # CFL条件に基づく時間刻み幅
        dx = state.levelset.dx
        return self.cfl * dx / (max_velocity + 1e-10)

    def step_forward(self, dt: float, **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める

        Args:
            dt: 時間刻み幅
            **kwargs: 追加のパラメータ
                - state: 現在のシミュレーション状態

        Returns:
            計算結果と診断情報を含む辞書
        """
        try:
            # 時間発展の実行
            result = super().step_forward(dt, **kwargs)
            levelset_new = result["state"]

            # 必要に応じて再初期化
            if levelset_new.need_reinit():
                levelset_new = reinitialize_levelset(
                    levelset_new,
                    dt=levelset_new.params.reinit_dt,
                    n_steps=levelset_new.params.reinit_steps,
                )

            # 体積保存の補正
            state = kwargs.get("state")
            if state is not None:
                initial_volume = state.levelset.compute_volume()
                current_volume = levelset_new.compute_volume()
                if abs(current_volume) > 1e-10:
                    levelset_new.data *= (initial_volume / current_volume) ** (
                        1.0 / levelset_new.ndim
                    )

            # 診断情報の収集
            diagnostics = {
                "time": self.time,
                "dt": dt,
                "reinitialized": levelset_new.need_reinit(),
                "volume": float(levelset_new.compute_volume()),
                "volume_error": float(
                    abs(
                        levelset_new.compute_volume() / state.levelset.compute_volume()
                        - 1.0
                    )
                ),
                "interface_area": float(levelset_new.compute_area()),
            }

            return {"levelset": levelset_new, "diagnostics": diagnostics}

        except Exception as e:
            if self.logger:
                self.logger.error(f"Level Setソルバーの時間発展中にエラー: {e}")
            raise

    def get_diagnostics(self) -> Dict[str, Any]:
        """ソルバーの診断情報を取得

        Returns:
            診断情報の辞書
        """
        diag = super().get_diagnostics()
        diag.update(
            {
                "weno_order": self.weno_order,
                "discretization_scheme": "WENO" if self.use_weno else "Central",
                "active_terms": len(self.terms),
            }
        )
        return diag
