"""WENOスキームの補間係数を管理するモジュール

このモジュールは、WENOスキームで使用される線形補間係数と
非線形重み係数の計算および管理を担当します。
"""

from typing import Dict, List, Tuple
import numpy as np
from functools import lru_cache


class WENOCoefficients:
    """WENOスキームの補間係数を管理するクラス"""

    def __init__(self):
        """WENO係数管理クラスを初期化"""
        self._coeff_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._optimal_weights_cache: Dict[int, np.ndarray] = {}

    @lru_cache(maxsize=8)
    def get_interpolation_coefficients(
        self, order: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """指定された次数のWENO補間係数を取得

        Args:
            order: WENOスキームの次数

        Returns:
            (補間係数, 理想重み係数)のタプル

        Raises:
            ValueError: 未対応の次数が指定された場合
        """
        if order not in [3, 5, 7]:
            raise ValueError(f"未対応のWENO次数です: {order}")

        if order not in self._coeff_cache:
            self._coeff_cache[order] = self._compute_coefficients(order)

        return self._coeff_cache[order]

    def _compute_coefficients(self, order: int) -> Tuple[np.ndarray, np.ndarray]:
        """WENO補間係数を計算

        Args:
            order: WENOスキームの次数

        Returns:
            (補間係数, 理想重み係数)のタプル
        """
        if order == 3:
            return self._compute_weno3_coefficients()
        elif order == 5:
            return self._compute_weno5_coefficients()
        else:  # order == 7
            return self._compute_weno7_coefficients()

    def _compute_weno3_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """WENO3の補間係数を計算"""
        # 3次精度WENOの補間係数（2つのステンシル）
        coeffs = np.array(
            [
                [-1 / 2, 3 / 2],  # ステンシル0の係数
                [1 / 2, 1 / 2],  # ステンシル1の係数
            ]
        )

        # 理想重み係数
        optimal_weights = np.array([1 / 3, 2 / 3])

        return coeffs, optimal_weights

    def _compute_weno5_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """WENO5の補間係数を計算"""
        # 5次精度WENOの補間係数（3つのステンシル）
        coeffs = np.array(
            [
                [1 / 3, -7 / 6, 11 / 6],  # ステンシル0の係数
                [-1 / 6, 5 / 6, 1 / 3],  # ステンシル1の係数
                [1 / 3, 5 / 6, -1 / 6],  # ステンシル2の係数
            ]
        )

        # 理想重み係数
        optimal_weights = np.array([0.1, 0.6, 0.3])

        return coeffs, optimal_weights

    def _compute_weno7_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """WENO7の補間係数を計算"""
        # 7次精度WENOの補間係数（4つのステンシル）
        coeffs = np.array(
            [
                [-1 / 4, 13 / 12, -23 / 12, 25 / 12],  # ステンシル0の係数
                [1 / 12, -5 / 12, 13 / 12, 1 / 4],  # ステンシル1の係数
                [-1 / 12, 7 / 12, 7 / 12, -1 / 12],  # ステンシル2の係数
                [1 / 4, 13 / 12, -5 / 12, 1 / 12],  # ステンシル3の係数
            ]
        )

        # 理想重み係数
        optimal_weights = np.array([1 / 20, 9 / 20, 9 / 20, 1 / 20])

        return coeffs, optimal_weights

    def get_smoothness_coefficients(self, order: int) -> List[np.ndarray]:
        """滑らかさ指標の計算に使用する係数を取得

        Args:
            order: WENOスキームの次数

        Returns:
            各ステンシルの滑らかさ指標計算用係数のリスト
        """
        if order == 3:
            return [
                np.array([[1, -2, 1]]),  # β_0
                np.array([[1, -2, 1]]),  # β_1
            ]
        elif order == 5:
            return [
                np.array([[1, -2, 1], [1, -4, 3]]),  # β_0
                np.array([[1, -2, 1], [1, -2, 1]]),  # β_1
                np.array([[1, -2, 1], [3, -4, 1]]),  # β_2
            ]
        elif order == 7:
            return [
                np.array([[1, -2, 1], [1, -4, 3], [1, -6, 5]]),  # β_0
                np.array([[1, -2, 1], [1, -4, 3], [1, -4, 3]]),  # β_1
                np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]]),  # β_2
                np.array([[1, -2, 1], [3, -4, 1], [5, -6, 1]]),  # β_3
            ]
        else:
            raise ValueError(f"未対応のWENO次数です: {order}")
