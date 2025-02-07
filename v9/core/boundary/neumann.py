"""ノイマン境界条件を提供するモジュール

このモジュールは、流体シミュレーションで使用されるノイマン境界条件を実装します。
ノイマン境界条件では、境界上で物理量の勾配を指定します。
"""

import numpy as np
from .base import BoundaryCondition, StencilInfo

class NeumannBoundary(BoundaryCondition):
    """ノイマン境界条件クラス
    
    ノイマン境界条件は、境界上で物理量の勾配を指定します。
    断熱壁や流出境界などで使用されます。
    """
    
    def __init__(self, gradient: float = 0.0, order: int = 2):
        """ノイマン境界条件を初期化
        
        Args:
            gradient: 境界での勾配
            order: 差分近似の次数
        """
        super().__init__(order)
        self.gradient = gradient
    
    def apply(self, field: np.ndarray, axis: int, side: int) -> np.ndarray:
        """ノイマン境界条件を適用
        
        Args:
            field: 境界条件を適用する場
            axis: 境界条件を適用する軸
            side: 境界の側（0: 負側、1: 正側）
            
        Returns:
            境界条件が適用された場
        """
        self.validate_field(field, axis)
        result = field.copy()
        dx = 1.0  # 正規化された格子間隔
        
        # 境界近傍の値を取得
        if side == 0:  # 負側の境界
            interior_slice = self.get_boundary_slice(field, axis, 1, 1)
            interior_value = field[interior_slice]
            # 勾配条件に基づいて境界値を設定
            boundary_slice = self.get_boundary_slice(field, axis, 0, 1)
            result[boundary_slice] = interior_value - self.gradient * dx
        else:  # 正側の境界
            interior_slice = self.get_boundary_slice(field, axis, 0, 1)
            interior_value = field[interior_slice]
            # 勾配条件に基づいて境界値を設定
            boundary_slice = self.get_boundary_slice(field, axis, 1, 1)
            result[boundary_slice] = interior_value + self.gradient * dx
        
        return result
    
    def get_stencil(self, side: int) -> StencilInfo:
        """差分ステンシルの情報を取得
        
        Args:
            side: 境界の側（0: 負側、1: 正側）
            
        Returns:
            ステンシルの情報
        """
        # 2次精度の場合
        if self.order == 2:
            if side == 0:
                return StencilInfo(
                    points=np.array([0, 1]),
                    coefficients=np.array([-1.0, 1.0])
                )
            else:
                return StencilInfo(
                    points=np.array([-1, 0]),
                    coefficients=np.array([-1.0, 1.0])
                )
        # 4次精度の場合
        elif self.order == 4:
            if side == 0:
                return StencilInfo(
                    points=np.array([0, 1, 2, 3]),
                    coefficients=np.array([-11/6, 3, -3/2, 1/3])
                )
            else:
                return StencilInfo(
                    points=np.array([-3, -2, -1, 0]),
                    coefficients=np.array([-1/3, 3/2, -3, 11/6])
                )
        else:
            raise ValueError(f"未対応の次数です: {self.order}")
