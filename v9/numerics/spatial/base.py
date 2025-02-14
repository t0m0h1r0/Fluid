"""空間微分スキームの基底クラスを提供するモジュール（改良版）"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from core.field import ScalarField


@dataclass
class DifferentiationConfig:
    """空間微分の設定クラス"""
    order: int = 2
    padding_mode: str = "reflect"
    boundary_order: Optional[int] = None

    def validate(self):
        """設定値の妥当性を検証"""
        if self.order not in [2, 4, 6]:
            raise ValueError("差分近似の次数は2, 4, 6のいずれかである必要があります")

        valid_padding_modes = ["reflect", "constant", "nearest", "mirror"]
        if self.padding_mode not in valid_padding_modes:
            raise ValueError(f"無効なpadding_mode。選択肢: {valid_padding_modes}")

        if self.boundary_order is not None and self.boundary_order > self.order:
            raise ValueError("境界での近似次数は内部の次数以下である必要があります")


class SpatialDerivative(ABC):
    """空間微分の基底クラス（改良版）"""

    def __init__(self, config: Optional[DifferentiationConfig] = None):
        """空間微分計算器を初期化

        Args:
            config: 微分計算の設定
        """
        self.config = config or DifferentiationConfig()
        self.config.validate()

    @abstractmethod
    def __call__(self, field: ScalarField, axis: int, dx: float) -> ScalarField:
        """空間微分を計算（改良版）

        Args:
            field: 入力スカラー場
            axis: 微分を計算する軸
            dx: グリッド間隔

        Returns:
            微分値を格納した新しいスカラー場
        """
        pass

    @abstractmethod
    def get_stencil(self) -> Tuple[np.ndarray, np.ndarray]:
        """差分ステンシルを取得

        Returns:
            (points, coefficients) - ステンシルの位置と係数のタプル
        """
        pass

    def _validate_input(self, field: ScalarField, axis: int, dx: float):
        """入力データの妥当性を検証

        Args:
            field: 入力スカラー場
            axis: 微分を計算する軸
            dx: グリッド間隔

        Raises:
            ValueError: 無効な入力が指定された場合
        """
        if not isinstance(field, ScalarField):
            raise ValueError("fieldはScalarFieldである必要があります")

        if not 0 <= axis < field.ndim:
            raise ValueError(f"無効な軸です: {axis}")

        if dx <= 0:
            raise ValueError("グリッド間隔は正の値である必要があります")

    def get_padding_width(self) -> int:
        """必要なパディング幅を取得

        Returns:
            パディングが必要な点の数
        """
        return self.config.order // 2

    def apply_padding(
        self, field: ScalarField, axis: int, width: Optional[int] = None
    ) -> ScalarField:
        """スカラー場にパディングを適用（改良版）

        Args:
            field: 入力スカラー場
            axis: パディングを適用する軸
            width: パディング幅（Noneの場合は次数から自動決定）

        Returns:
            パディングが適用された新しいスカラー場
        """
        width = width or self.get_padding_width()
        # 新しい演算子を活用したパディング
        return field.pad(axis, width, mode=self.config.padding_mode)

    def get_boundary_stencil(self, side: int) -> Tuple[np.ndarray, np.ndarray]:
        """境界での差分ステンシルを取得（改良版）

        Args:
            side: 境界の側（0: 負側、1: 正側）

        Returns:
            (points, coefficients) - 境界ステンシルの位置と係数のタプル
        """
        return self.get_stencil()

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得（改良版）"""
        return {
            "order": self.config.order,
            "padding_mode": self.config.padding_mode,
            "boundary_order": self.config.boundary_order,
            "stencil_points": self.get_stencil()[0].tolist(),
            "stencil_coefficients": self.get_stencil()[1].tolist(),
        }