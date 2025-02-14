"""中心差分スキームを提供するモジュール

このモジュールは、2次、4次、6次の中心差分スキームを実装します。
"""

from typing import Dict, Any
from ..base import SpatialDerivative, DifferentiationConfig
from ..stencil import DifferenceStencils, StencilCoefficients
from core.field import ScalarField


class CentralDifference(SpatialDerivative):
    """中心差分による空間微分計算クラス

    2次、4次、6次の中心差分スキームを提供します。
    """

    def __init__(
        self,
        config: DifferentiationConfig = None,
        derivative_order: int = 1,
    ):
        """中心差分計算器を初期化

        Args:
            config: 差分計算の設定
            derivative_order: 微分の階数（1または2）
        """
        super().__init__(config)
        if derivative_order not in [1, 2]:
            raise ValueError("微分の階数は1または2である必要があります")
        self.derivative_order = derivative_order

    def __call__(self, field: ScalarField, axis: int, dx: float) -> ScalarField:
        """中心差分により微分を計算

        Args:
            field: 入力スカラー場
            axis: 微分を計算する軸
            dx: グリッド間隔

        Returns:
            微分値を格納した新しいスカラー場
        """
        self._validate_input(field, axis, dx)

        # パディングの適用
        padded_field = self.apply_padding(field, axis)

        # ステンシルの取得と適用
        stencil = self.get_stencil()
        derivative = DifferenceStencils.apply_stencil(padded_field, stencil, axis, dx)

        # パディング部分の除去
        result = derivative.trim([axis], [self.get_padding_width()])

        return result

    def get_stencil(self) -> StencilCoefficients:
        """差分ステンシルを取得

        Returns:
            ステンシル係数
        """
        if self.derivative_order == 1:
            return DifferenceStencils.get_first_derivative_stencil(self.config.order)
        else:  # derivative_order == 2
            return DifferenceStencils.get_second_derivative_stencil(self.config.order)

    def get_boundary_stencil(self, side: int) -> StencilCoefficients:
        """境界での差分ステンシルを取得

        Args:
            side: 境界の側（0: 負側、1: 正側）

        Returns:
            境界用のステンシル係数
        """
        if self.derivative_order == 1:
            order = self.config.boundary_order or self.config.order
            return DifferenceStencils.get_first_derivative_stencil(
                order, boundary=True, side=side
            )
        else:
            # 2階微分の場合は中心差分を使用
            return self.get_stencil()

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得

        Returns:
            診断情報の辞書
        """
        diag = super().get_diagnostics()
        diag.update(
            {
                "derivative_order": self.derivative_order,
                "method": "central",
                "stencil_width": self.get_padding_width() * 2 + 1,
            }
        )
        return diag
