"""3次精度WENOスキームの実装

このモジュールは、3次精度のWeighted Essentially Non-Oscillatory (WENO3)
スキームを実装します。
"""

import numpy as np
import numpy.typing as npt
from typing import List, Dict, Any

from ..base import WENOBase
from ..coefficients import WENOCoefficients
from ..smoothness import SmoothnessIndicator
from ..weights import WeightCalculator


class WENO3(WENOBase):
    """3次精度WENOスキームのクラス"""

    def __init__(self, epsilon: float = 1e-6, p: float = 2.0):
        """WENO3スキームを初期化

        Args:
            epsilon: ゼロ除算を防ぐための小さな値
            p: 非線形重みの指数
        """
        super().__init__(order=3, epsilon=epsilon)

        # 係数と計算機の初期化
        self._coeffs = WENOCoefficients()
        self._smoother = SmoothnessIndicator()
        self._weight_calc = WeightCalculator(epsilon=epsilon, p=p)

        # 補間係数と理想重み係数の取得
        self._interpolation_coeffs, self._optimal_weights = (
            self._coeffs.get_interpolation_coefficients(3)
        )

    def reconstruct(
        self, data: npt.NDArray[np.float64], axis: int = -1
    ) -> npt.NDArray[np.float64]:
        """WENO3による再構成を実行

        Args:
            data: 入力データ配列
            axis: 再構成を行う軸

        Returns:
            再構成された値の配列
        """
        # 入力の検証
        self._validate_input(data, axis)

        # 滑らかさ指標の計算
        smoothness_coeffs = self._coeffs.get_smoothness_coefficients(3)
        beta = self._smoother.compute(data, smoothness_coeffs, axis)

        # 非線形重み係数の計算
        omega, _ = self._weight_calc.compute_weights(beta, self._optimal_weights)

        # 各ステンシルの寄与を計算
        result = np.zeros_like(data)

        # 各ステンシルについて
        for k in range(len(omega)):
            # このステンシルの補間値を計算
            stencil_value = np.zeros_like(data)
            for j, coeff in enumerate(self._interpolation_coeffs[k]):
                stencil_value += coeff * np.roll(data, j - 1, axis=axis)

            # 重み付きで加算
            result += omega[k] * stencil_value

        return result

    def compute_smoothness_indicators(
        self, data: npt.NDArray[np.float64], axis: int = -1
    ) -> List[npt.NDArray[np.float64]]:
        """滑らかさ指標を計算

        Args:
            data: 入力データ配列
            axis: 計算を行う軸

        Returns:
            各ステンシルの滑らかさ指標
        """
        smoothness_coeffs = self._coeffs.get_smoothness_coefficients(3)
        return self._smoother.compute(data, smoothness_coeffs, axis)

    def get_status(self) -> Dict[str, Any]:
        """WENOスキームの状態を取得"""
        status = super().get_status()
        status.update(
            {
                "coefficients": {
                    "interpolation": self._interpolation_coeffs.tolist(),
                    "optimal_weights": self._optimal_weights.tolist(),
                },
                "weight_calculator": self._weight_calc.get_diagnostics(),
            }
        )
        return status
