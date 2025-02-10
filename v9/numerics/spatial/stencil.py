"""差分ステンシルを定義するモジュール

このモジュールは、各種の差分スキームで使用されるステンシルを定義します。
2次、4次、6次の中心差分と、境界での特殊なステンシルを提供します。
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class StencilCoefficients:
    """差分ステンシルの係数を保持するクラス

    Attributes:
        points: ステンシル点の相対位置
        coefficients: 各点での係数
    """

    points: np.ndarray
    coefficients: np.ndarray

    def validate(self):
        """ステンシル係数の妥当性を検証"""
        if len(self.points) != len(self.coefficients):
            raise ValueError("点の数と係数の数が一致しません")


class DifferenceStencils:
    """差分ステンシルの定義を提供するクラス"""

    # 1階微分の中心差分ステンシル
    CENTRAL_FIRST = {
        2: StencilCoefficients(  # 2次精度
            points=np.array([-1, 0, 1]),
            coefficients=np.array([-1 / 2, 0, 1 / 2]),
        ),
        4: StencilCoefficients(  # 4次精度
            points=np.array([-2, -1, 0, 1, 2]),
            coefficients=np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
        ),
        6: StencilCoefficients(  # 6次精度
            points=np.array([-3, -2, -1, 0, 1, 2, 3]),
            coefficients=np.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]),
        ),
    }

    # 2階微分の中心差分ステンシル
    CENTRAL_SECOND = {
        2: StencilCoefficients(  # 2次精度
            points=np.array([-1, 0, 1]),
            coefficients=np.array([1, -2, 1]),
        ),
        4: StencilCoefficients(  # 4次精度
            points=np.array([-2, -1, 0, 1, 2]),
            coefficients=np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
        ),
        6: StencilCoefficients(  # 6次精度
            points=np.array([-3, -2, -1, 0, 1, 2, 3]),
            coefficients=np.array(
                [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
            ),
        ),
    }

    # 境界での1階微分ステンシル（前方差分）
    FORWARD_FIRST = {
        2: StencilCoefficients(  # 2次精度
            points=np.array([0, 1, 2]),
            coefficients=np.array([-3 / 2, 2, -1 / 2]),
        ),
        4: StencilCoefficients(  # 4次精度
            points=np.array([0, 1, 2, 3, 4]),
            coefficients=np.array([-25 / 12, 4, -3, 4 / 3, -1 / 4]),
        ),
    }

    # 境界での1階微分ステンシル（後方差分）
    BACKWARD_FIRST = {
        2: StencilCoefficients(  # 2次精度
            points=np.array([-2, -1, 0]),
            coefficients=np.array([1 / 2, -2, 3 / 2]),
        ),
        4: StencilCoefficients(  # 4次精度
            points=np.array([-4, -3, -2, -1, 0]),
            coefficients=np.array([1 / 4, -4 / 3, 3, -4, 25 / 12]),
        ),
    }

    @classmethod
    def get_first_derivative_stencil(
        cls, order: int, boundary: bool = False, side: int = 0
    ) -> StencilCoefficients:
        """1階微分のステンシルを取得

        Args:
            order: 精度次数
            boundary: 境界用のステンシルかどうか
            side: 境界の側（0: 負側、1: 正側）

        Returns:
            ステンシル係数
        """
        if not boundary:
            if order not in cls.CENTRAL_FIRST:
                raise ValueError(f"未対応の次数です: {order}")
            return cls.CENTRAL_FIRST[order]
        else:
            if order not in cls.FORWARD_FIRST:
                raise ValueError(f"境界では未対応の次数です: {order}")
            if side == 0:
                return cls.FORWARD_FIRST[order]
            else:
                return cls.BACKWARD_FIRST[order]

    @classmethod
    def get_second_derivative_stencil(cls, order: int) -> StencilCoefficients:
        """2階微分のステンシルを取得

        Args:
            order: 精度次数

        Returns:
            ステンシル係数
        """
        if order not in cls.CENTRAL_SECOND:
            raise ValueError(f"未対応の次数です: {order}")
        return cls.CENTRAL_SECOND[order]

    @staticmethod
    def apply_stencil(
        data: np.ndarray, stencil: StencilCoefficients, axis: int, dx: float
    ) -> np.ndarray:
        """ステンシルを適用して微分を計算

        Args:
            data: 入力データ
            stencil: 適用するステンシル
            axis: 微分を計算する軸
            dx: グリッド間隔

        Returns:
            計算された微分
        """
        result = np.zeros_like(data)
        for point, coef in zip(stencil.points, stencil.coefficients):
            result += coef * np.roll(data, point, axis=axis)
        return result / (dx ** len(stencil.points))
