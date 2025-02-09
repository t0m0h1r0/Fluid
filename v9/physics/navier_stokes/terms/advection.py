"""移流項を実装するモジュール

このモジュールは、Navier-Stokes方程式の移流項 (u・∇)u を実装します。
WENOスキームによる高精度な空間離散化を提供します。
"""

from typing import List, Dict, Any, Optional
import numpy as np
from core.field import VectorField
from ..base import NavierStokesTerm


class AdvectionTerm(NavierStokesTerm):
    """移流項クラス"""

    def __init__(self, use_weno: bool = True, weno_order: int = 5):
        """移流項を初期化

        Args:
            use_weno: WENOスキームを使用するかどうか
            weno_order: WENOスキームの次数（3または5）
        """
        self._name = "Advection"
        self.use_weno = use_weno
        self.weno_order = weno_order

        # WENOスキームの係数を初期化
        if use_weno:
            self._init_weno_coefficients()

    @property
    def name(self) -> str:
        """項の名前を取得"""
        return self._name

    def _init_weno_coefficients(self):
        """WENOスキームの係数を初期化"""
        # WENO5の場合の係数
        if self.weno_order == 5:
            # 線形重み
            self.weno_weights = np.array([0.1, 0.6, 0.3])

            # 各ステンシルでの係数
            self.weno_coeffs = [
                np.array([1 / 3, -7 / 6, 11 / 6]),  # 左側ステンシル
                np.array([-1 / 6, 5 / 6, 1 / 3]),  # 中央ステンシル
                np.array([1 / 3, 5 / 6, -1 / 6]),  # 右側ステンシル
            ]

        # WENO3の場合の係数
        elif self.weno_order == 3:
            self.weno_weights = np.array([1 / 3, 2 / 3])
            self.weno_coeffs = [
                np.array([-1 / 2, 3 / 2]),  # 左側ステンシル
                np.array([1 / 2, 1 / 2]),  # 右側ステンシル
            ]
        else:
            raise ValueError(f"未対応のWENO次数です: {self.weno_order}")

    def _weno_reconstruction(self, values: np.ndarray, axis: int) -> np.ndarray:
        """WENOスキームによる再構築

        Args:
            values: 再構築する値の配列
            axis: 再構築を行う軸

        Returns:
            再構築された値
        """
        # シフトしたインデックスでの値を取得
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

            # 非線形重みを計算（ブロードキャストを考慮）
            weights = self.weno_weights.reshape((-1,) + (1,) * values.ndim)
            alpha0 = weights[0] / (eps + beta0) ** 2
            alpha1 = weights[1] / (eps + beta1) ** 2
            alpha2 = weights[2] / (eps + beta2) ** 2
            alpha_sum = alpha0 + alpha1 + alpha2

            omega0 = alpha0 / alpha_sum
            omega1 = alpha1 / alpha_sum
            omega2 = alpha2 / alpha_sum

            # 各ステンシルでの補間値を計算
            p0 = (
                self.weno_coeffs[0][0] * v1
                + self.weno_coeffs[0][1] * v2
                + self.weno_coeffs[0][2] * v3
            )

            p1 = (
                self.weno_coeffs[1][0] * v2
                + self.weno_coeffs[1][1] * v3
                + self.weno_coeffs[1][2] * v4
            )

            p2 = (
                self.weno_coeffs[2][0] * v3
                + self.weno_coeffs[2][1] * v4
                + self.weno_coeffs[2][2] * v5
            )

            return omega0 * p0 + omega1 * p1 + omega2 * p2

        else:  # WENO3
            v1 = np.roll(values, 1, axis=axis)
            v2 = values
            v3 = np.roll(values, -1, axis=axis)

            beta0 = (v2 - v1) ** 2
            beta1 = (v3 - v2) ** 2

            eps = 1e-6
            weights = self.weno_weights.reshape((-1,) + (1,) * values.ndim)
            alpha0 = weights[0] / (eps + beta0) ** 2
            alpha1 = weights[1] / (eps + beta1) ** 2
            alpha_sum = alpha0 + alpha1

            omega0 = alpha0 / alpha_sum
            omega1 = alpha1 / alpha_sum

            p0 = self.weno_coeffs[0][0] * v1 + self.weno_coeffs[0][1] * v2
            p1 = self.weno_coeffs[1][0] * v2 + self.weno_coeffs[1][1] * v3

            return omega0 * p0 + omega1 * p1

    def compute(
        self, 
        velocity: VectorField, 
        state: Optional[Any] = None, 
        dt: Optional[float] = None, 
        **kwargs
    ) -> List[np.ndarray]:
        """移流項の寄与を計算

        Args:
            velocity: 現在の速度場
            state: シミュレーション状態（オプション）
            dt: 時間刻み幅（オプション）
            **kwargs: 追加のパラメータ

        Returns:
            各方向の速度成分への寄与のリスト
        """
        result = []

        if self.use_weno:
            # WENOスキームによる空間離散化
            for i, v_i in enumerate(velocity.components):
                flux = np.zeros_like(v_i.data)
                for j, v_j in enumerate(velocity.components):
                    # 風上差分の方向を決定
                    upwind = v_j.data < 0

                    # 正の速度に対する flux
                    v_plus = self._weno_reconstruction(v_i.data, j)
                    # 負の速度に対する flux
                    v_minus = self._weno_reconstruction(np.flip(v_i.data, j), j)
                    v_minus = np.flip(v_minus, j)

                    # 風上方向に応じてfluxを選択
                    flux += v_j.data * np.where(upwind, v_minus, v_plus)

                result.append(-flux)

        else:
            # 標準的な中心差分
            for i, v_i in enumerate(velocity.components):
                result.append(
                    -sum(
                        v_j.data * v_i.gradient(j)
                        for j, v_j in enumerate(velocity.components)
                    )
                )

        return result

    def get_diagnostics(self, velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """移流項の診断情報を取得"""
        # 移流量の計算
        flux = self.compute(velocity, 1.0)
        max_flux = max(np.max(np.abs(f)) for f in flux)
        mean_flux = np.mean([np.mean(np.abs(f)) for f in flux])

        return {
            "scheme": "WENO" if self.use_weno else "central",
            "weno_order": self.weno_order if self.use_weno else None,
            "max_flux": float(max_flux),
            "mean_flux": float(mean_flux),
        }
