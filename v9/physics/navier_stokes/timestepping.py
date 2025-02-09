"""時間積分スキームを提供するモジュール

このモジュールは、Navier-Stokes方程式の時間積分に使用される
様々な時間積分スキームを実装します。
"""

from typing import List
import numpy as np
from core.field import VectorField
from .base import TimeIntegrator, NavierStokesTerm


class ForwardEuler(TimeIntegrator):
    """前進Euler法による時間積分

    最も単純な1次精度の時間積分スキーム。
    """

    def step(
        self,
        initial_state: VectorField,
        dt: float,
        terms: List[NavierStokesTerm],
        **kwargs,
    ) -> VectorField:
        """1時間ステップを実行

        Args:
            initial_state: 初期状態
            dt: 時間刻み幅
            terms: NS方程式の各項
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        result = initial_state.copy()

        # 各項の寄与を計算して足し合わせる
        for term in terms:
            contributions = term.compute(initial_state, dt, **kwargs)
            for i, contribution in enumerate(contributions):
                result.components[i].data += dt * contribution

        return result


class RungeKutta4(TimeIntegrator):
    """4次のRunge-Kutta法による時間積分

    4次精度の古典的なRunge-Kutta法を実装します。
    """

    def step(
        self,
        initial_state: VectorField,
        dt: float,
        terms: List[NavierStokesTerm],
        **kwargs,
    ) -> VectorField:
        """1時間ステップを実行

        Args:
            initial_state: 初期状態
            dt: 時間刻み幅
            terms: NS方程式の各項
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        # k1の計算
        k1 = self._compute_total_contribution(initial_state, terms, **kwargs)

        # k2の計算
        state_k2 = initial_state.copy()
        for i, comp in enumerate(state_k2.components):
            comp.data += 0.5 * dt * k1[i]
        k2 = self._compute_total_contribution(state_k2, terms, **kwargs)

        # k3の計算
        state_k3 = initial_state.copy()
        for i, comp in enumerate(state_k3.components):
            comp.data += 0.5 * dt * k2[i]
        k3 = self._compute_total_contribution(state_k3, terms, **kwargs)

        # k4の計算
        state_k4 = initial_state.copy()
        for i, comp in enumerate(state_k4.components):
            comp.data += dt * k3[i]
        k4 = self._compute_total_contribution(state_k4, terms, **kwargs)

        # 最終的な更新
        result = initial_state.copy()
        for i, comp in enumerate(result.components):
            comp.data += (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])

        return result

    def _compute_total_contribution(
        self, state: VectorField, terms: List[NavierStokesTerm], **kwargs
    ) -> List[np.ndarray]:
        """全ての項からの寄与を計算"""
        result = [np.zeros_like(comp.data) for comp in state.components]

        for term in terms:
            contributions = term.compute(state, 1.0, **kwargs)  # dtは後で考慮
            for i, contribution in enumerate(contributions):
                result[i] += contribution

        return result
