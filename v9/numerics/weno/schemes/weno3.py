import numpy as np
import numpy.typing as npt
from typing import List

from ..base import WENOBase
from ..coefficients import WENOCoefficients
from ..smoothness import SmoothnessIndicator
from ..weights import WeightCalculator


class WENO3(WENOBase):
    """3次精度WENOスキームのクラス（演算子改善版）"""

    def __init__(self, epsilon: float = 1e-6, p: float = 2.0):
        super().__init__(order=3, epsilon=epsilon)
        self._coeffs = WENOCoefficients()
        self._smoother = SmoothnessIndicator()
        self._weight_calc = WeightCalculator(epsilon=epsilon, p=p)
        self._interpolation_coeffs, self._optimal_weights = (
            self._coeffs.get_interpolation_coefficients(3)
        )

    def reconstruct(
        self, field: npt.NDArray[np.float64], axis: int = -1
    ) -> npt.NDArray[np.float64]:
        """WENO3による再構成を実行（新しい演算子を活用）"""
        self._validate_input(field, axis)

        # 滑らかさ指標の計算（新しい演算子を活用）
        smoothness_coeffs = self._coeffs.get_smoothness_coefficients(3)
        beta = self._smoother.compute(field, smoothness_coeffs, axis)

        # 非線形重み係数の計算
        omega, _ = self._weight_calc.compute_weights(beta, self._optimal_weights)

        # 各ステンシルの寄与を計算（新しい演算子を活用）
        result = np.zeros_like(field)
        for k in range(len(omega)):
            # このステンシルの補間値を計算
            stencil_value = np.zeros_like(field)
            for j, coeff in enumerate(self._interpolation_coeffs[k]):
                stencil_value += coeff * np.roll(field, j - 1, axis=axis)
            # 重み付き加算（新しい * 演算子を活用）
            result += omega[k] * stencil_value

        return result

    def compute_smoothness_indicators(
        self, field: npt.NDArray[np.float64], axis: int = -1
    ) -> List[npt.NDArray[np.float64]]:
        """滑らかさ指標を計算（新しい演算子を活用）"""
        smoothness_coeffs = self._coeffs.get_smoothness_coefficients(3)
        return self._smoother.compute(field, smoothness_coeffs, axis)
