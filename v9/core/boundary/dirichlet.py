"""ディリクレ境界条件を提供するモジュール

このモジュールは、流体シミュレーションで使用されるディリクレ境界条件を実装します。
ディリクレ境界条件では、境界上で物理量の値を指定します。
"""

import numpy as np
from .base import BoundaryCondition, StencilInfo

class DirichletBoundary(BoundaryCondition):
    """ディリクレ境界条件クラス
    
    ディリクレ境界条件は、境界上で物理量の値を指定します。
    壁面での速度や温度などを指定する場合に使用されます。
    """
    
    def __init__(self, value: float = 0.0, order: int = 2):
        """ディリクレ境界条件を初期化
        
        Args:
            value: 境界での値
            order: 差分近似の次数
        """
        super().__init__(order)
        self.value = value
    
    def apply(self, field: np.ndarray, axis: int, side: int) -> np.ndarray:
        """ディリクレ境界条件を適用
        
        Args:
            field: 境界条件を適用する場
            axis: 境界条件を適用する軸
            side: 境界の側（0: 負側、1: 正側）
            
        Returns:
            境界条件が適用された場
        """
        self.validate_field(field, axis)
        result = field.copy()
        
        # 境界面のスライスを取得
        boundary_slice = self.get_boundary_slice(field, axis, side, 1)
        
        # 境界値を設定
        result[boundary_slice] = self.value
        
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
                    points=np.array([0, 1, 2]),
                    coefficients=np.array([-3/2, 2, -1/2])
                )
            else:
                return StencilInfo(
                    points=np.array([-2, -1, 0]),
                    coefficients=np.array([1/2, -2, 3/2])
                )
        # 4次精度の場合
        elif self.order == 4:
            if side == 0:
                return StencilInfo(
                    points=np.array([0, 1, 2, 3, 4]),
                    coefficients=np.array([-25/12, 4, -3, 4/3, -1/4])
                )
            else:
                return StencilInfo(
                    points=np.array([-4, -3, -2, -1, 0]),
                    coefficients=np.array([1/4, -4/3, 3, -4, 25/12])
                )
        else:
            raise ValueError(f"未対応の次数です: {self.order}")
