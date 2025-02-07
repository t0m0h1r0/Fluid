"""保存則を持つ場のクラスを提供するモジュール

このモジュールは、保存則を持つ場（質量、運動量など）の基底クラスを定義します。
"""

from typing import Dict, Any
import numpy as np
from .field import Field

class ConservedField(Field):
    """保存則を持つ場の基底クラス
    
    この基底クラスは、保存則を持つ物理量（質量、運動量など）の場に共通の
    機能を提供します。初期状態の積分値を保持し、保存則の検証を可能にします。

    Attributes:
        initial_integral (float): 初期状態での積分値
    """
    
    def __init__(self, *args, **kwargs):
        """保存則を持つ場を初期化
        
        Args:
            *args: Field.__init__に渡される位置引数
            **kwargs: Field.__init__に渡されるキーワード引数
        """
        super().__init__(*args, **kwargs)
        self._initial_integral = self.integrate()
    
    def integrate(self) -> float:
        """場の積分値を計算
        
        空間全体にわたる場の積分値を計算します。
        
        Returns:
            計算された積分値
        """
        return np.sum(self._data) * self._dx**self.ndim
    
    def check_conservation(self) -> float:
        """保存則の確認
        
        現在の積分値と初期積分値を比較し、相対誤差を計算します。
        
        Returns:
            保存則の相対誤差
        """
        current_integral = self.integrate()
        if abs(self._initial_integral) < 1e-10:  # ゼロ除算を防ぐ
            return 0.0 if abs(current_integral) < 1e-10 else float('inf')
        return abs(current_integral - self._initial_integral) / abs(self._initial_integral)
    
    def reset_reference(self):
        """参照積分値をリセット
        
        現在の状態を新しい参照状態として設定します。
        """
        self._initial_integral = self.integrate()
    
    def save_state(self) -> Dict[str, Any]:
        """現在の状態を保存
        
        Returns:
            現在の状態を表す辞書
        """
        state = super().save_state()
        state['initial_integral'] = self._initial_integral
        return state
    
    def load_state(self, state: Dict[str, Any]):
        """状態を読み込み
        
        Args:
            state: 読み込む状態の辞書
        """
        super().load_state(state)
        self._initial_integral = state['initial_integral']
    
    @property
    def conservation_error(self) -> float:
        """保存則の誤差を取得"""
        return self.check_conservation()
    
    @property
    def initial_integral(self) -> float:
        """初期積分値を取得"""
        return self._initial_integral