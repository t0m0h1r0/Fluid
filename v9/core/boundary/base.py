"""境界条件の基底クラスを提供するモジュール

このモジュールは、流体シミュレーションで使用される境界条件の基底クラスを定義します。
すべての具体的な境界条件（周期境界、ディリクレ境界など）は、この基底クラスを継承します。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np

@dataclass
class StencilInfo:
    """差分ステンシルの情報を保持するクラス
    
    Attributes:
        points: ステンシルの位置（中心からの相対位置）
        coefficients: 各点での係数
    """
    points: np.ndarray      # 形状: (N,)
    coefficients: np.ndarray  # 形状: (N,)

class BoundaryCondition(ABC):
    """境界条件の基底クラス
    
    この抽象基底クラスは、すべての境界条件に共通のインターフェースを定義します。
    """
    
    def __init__(self, order: int = 2):
        """境界条件を初期化
        
        Args:
            order: 差分近似の次数（デフォルトは2次精度）
        """
        self.order = order
    
    @abstractmethod
    def apply(self, field: np.ndarray, axis: int, side: int) -> np.ndarray:
        """境界条件を適用
        
        Args:
            field: 境界条件を適用する場
            axis: 境界条件を適用する軸
            side: 境界の側（0: 負側、1: 正側）
            
        Returns:
            境界条件が適用された場
        """
        pass
    
    @abstractmethod
    def get_stencil(self, side: int) -> StencilInfo:
        """差分ステンシルの情報を取得
        
        Args:
            side: 境界の側（0: 負側、1: 正側）
            
        Returns:
            ステンシルの情報
        """
        pass
    
    def validate_field(self, field: np.ndarray, axis: int) -> None:
        """場の妥当性をチェック
        
        Args:
            field: チェックする場
            axis: チェックする軸
            
        Raises:
            ValueError: 無効な場や軸が指定された場合
        """
        if not isinstance(field, np.ndarray):
            raise ValueError("fieldはnumpy配列である必要があります")
        if not 0 <= axis < field.ndim:
            raise ValueError(f"無効な軸です: {axis}")
    
    def apply_all(self, field: np.ndarray, axis: int) -> np.ndarray:
        """両側の境界に境界条件を適用
        
        Args:
            field: 境界条件を適用する場
            axis: 境界条件を適用する軸
            
        Returns:
            境界条件が適用された場
        """
        self.validate_field(field, axis)
        result = field.copy()
        
        # 負側の境界に適用
        result = self.apply(result, axis, 0)
        
        # 正側の境界に適用
        result = self.apply(result, axis, 1)
        
        return result
    
    def get_boundary_slice(self, field: np.ndarray, axis: int, 
                          side: int, width: int) -> Tuple[slice, ...]:
        """境界領域のスライスを取得
        
        Args:
            field: 対象の場
            axis: 境界条件を適用する軸
            side: 境界の側（0: 負側、1: 正側）
            width: 境界領域の幅
            
        Returns:
            境界領域を選択するスライスのタプル
        """
        slices = [slice(None)] * field.ndim
        if side == 0:
            slices[axis] = slice(0, width)
        else:
            slices[axis] = slice(-width, None)
        return tuple(slices)
    
    def get_ghost_points(self, field: np.ndarray, axis: int,
                        side: int) -> np.ndarray:
        """ゴースト点の座標を取得
        
        Args:
            field: 対象の場
            axis: 境界条件を適用する軸
            side: 境界の側（0: 負側、1: 正側）
            
        Returns:
            ゴースト点の座標配列
        """
        # ステンシル情報から必要なゴースト点の数を決定
        stencil = self.get_stencil(side)
        n_ghost = len(stencil.points)
        
        # 境界に沿った座標グリッドを生成
        shape = list(field.shape)
        shape[axis] = n_ghost
        coordinates = np.empty(shape + [field.ndim])
        
        # 各次元の座標を設定
        for dim in range(field.ndim):
            if dim == axis:
                if side == 0:
                    coords = np.arange(-n_ghost, 0)
                else:
                    coords = np.arange(field.shape[axis], 
                                     field.shape[axis] + n_ghost)
            else:
                coords = np.arange(field.shape[dim])
            coordinates[..., dim] = coords
        
        return coordinates