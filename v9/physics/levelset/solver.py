"""Level Set法のソルバーを提供するモジュール

このモジュールは、Level Set方程式の時間発展を計算するためのソルバーを提供します。
"""

from typing import Dict, Any, Optional, List
import numpy as np

from .base import LevelSetSolverBase, LevelSetTerm
from .field import LevelSetField
from .utils import extend_velocity
from .reinitialize import reinitialize_levelset


class LevelSetSolver(LevelSetSolverBase):
    """Level Set法のソルバークラス"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        terms: Optional[List[LevelSetTerm]] = None,
        logger=None,
        **kwargs  # 追加のキーワード引数を許可
    ):
        """ソルバーを初期化

        Args:
            config: ソルバー設定
            terms: Level Set方程式の各項
            logger: ロガー
            **kwargs: 追加のキーワード引数
        """
        # デフォルト設定
        config = config or {}

        # キーワード引数から空間離散化パラメータを抽出
        use_weno = kwargs.get('use_weno', config.get('discretization', {}).get('scheme', 'weno') == 'weno')
        weno_order = kwargs.get('weno_order', config.get('discretization', {}).get('order', 5))

        # 基底クラスの初期化
        super().__init__(
            use_weno=use_weno,
            weno_order=weno_order,
            logger=logger
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

        # 結果を格納するLevel Set場
        result = LevelSetField(
            data=np.zeros_like(state.data), dx=state.dx, params=state.params
        )

        if self.use_weno:
            # WENOスキームによる空間離散化
            flux = np.zeros_like(state.data)
            for i, v in enumerate(velocity.components):
                # 風上差分の方向を決定
                upwind = v.data < 0

                # 正の速度に対する flux
                phi_plus = self._compute_weno_reconstruction(state.data, i)
                # 負の速度に対する flux
                phi_minus = self._compute_weno_reconstruction(np.flip(state.data, i), i)
                phi_minus = np.flip(phi_minus, i)

                # 風上方向に応じてfluxを選択
                flux += v.data * np.where(upwind, phi_minus, phi_plus)

            result.data = -flux

        else:
            # 標準的な中心差分
            flux = sum(
                v.data * state.gradient(i) for i, v in enumerate(velocity.components)
            )
            result.data = -flux

        # 追加の項の寄与を計算
        for term in self.terms:
            if term.enabled:
                result.data += term.compute(state.data, velocity.data)

        return result

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
        dx = state.dx
        return 0.5 * dx / (max_velocity + 1e-10)

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
            state = kwargs.get("state")
            if state is None:
                raise ValueError("stateが指定されていません")

            # 速度場の拡張
            velocity = kwargs.get("velocity")
            if velocity is not None:
                velocity_extended = extend_velocity(velocity.data, state.data, state.dx)
            else:
                velocity_extended = None

            # 時間発展の実行
            derivative = self.compute_derivative(
                state, velocity=velocity, velocity_data=velocity_extended
            )

            # 新しいLevel Set場の計算
            levelset_new = LevelSetField(
                data=state.data + dt * derivative.data, dx=state.dx, params=state.params
            )

            # 再初期化の検討
            if levelset_new.need_reinit():
                levelset_new = reinitialize_levelset(
                    levelset_new,
                    dt=levelset_new.params.reinit_dt,
                    n_steps=levelset_new.params.reinit_steps,
                )

            # 体積保存の補正
            initial_volume = state.compute_volume()
            current_volume = levelset_new.compute_volume()
            if abs(current_volume) > 1e-10:
                levelset_new.data *= (initial_volume / current_volume) ** (
                    1.0 / levelset_new.ndim
                )

            # 診断情報の収集
            diagnostics = {
                "time": self._time + dt,
                "dt": dt,
                "reinitialized": levelset_new.need_reinit(),
                "volume": float(levelset_new.compute_volume()),
                "volume_error": float(
                    abs(levelset_new.compute_volume() / initial_volume - 1.0)
                ),
                "interface_area": float(levelset_new.compute_area()),
            }

            return {
                "levelset": levelset_new,
                "time": self._time + dt,
                "diagnostics": diagnostics,
            }

        except Exception as e:
            if self.logger:
                self.logger.error(f"Level Setソルバーの時間発展中にエラー: {e}")
            raise

    def get_diagnostics(self) -> Dict[str, Any]:
        """ソルバーの診断情報を取得

        Returns:
            診断情報の辞書
        """
        diag = {
            "weno_order": self.weno_order,
            "discretization_scheme": "WENO" if self.use_weno else "Central",
            "active_terms": len(self.terms),
        }

        # 各項の診断情報を追加
        term_diagnostics = {}
        for term in self.terms:
            if term.enabled:
                term_diagnostics[term.name] = term.get_diagnostics()

        if term_diagnostics:
            diag["terms"] = term_diagnostics

        return diag