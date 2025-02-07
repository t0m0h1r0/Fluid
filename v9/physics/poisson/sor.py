"""SOR法によるPoissonソルバーを提供するモジュール

このモジュールは、Successive Over-Relaxation (SOR)法によるPoisson方程式の
ソルバーを実装します。最適な緩和係数を自動的に推定する機能も提供します。
"""

import numpy as np
from typing import Dict, Any
from .solver import PoissonSolver

class SORSolver(PoissonSolver):
    """SOR法によるPoissonソルバー
    
    ガウス・ザイデル法に緩和係数を導入することで、収束を加速します。
    赤黒順序付けによる並列化可能な実装も提供します。
    """
    
    def __init__(self,
                 omega: float = 1.5,
                 use_redblack: bool = True,
                 auto_tune: bool = True,
                 **kwargs):
        """SORソルバーを初期化
        
        Args:
            omega: 緩和係数（1 < ω < 2）
            use_redblack: 赤黒順序付けを使用するかどうか
            auto_tune: 緩和係数を自動調整するかどうか
            **kwargs: 基底クラスに渡すパラメータ
        """
        super().__init__(**kwargs)
        self.omega = omega
        self.use_redblack = use_redblack
        self.auto_tune = auto_tune
        
        # 自動調整用のパラメータ
        self._spectral_radius = None
        self._update_interval = 10
        self._previous_diff = None
    
    def initialize(self, **kwargs) -> None:
        """ソルバーの初期化
        
        Args:
            **kwargs: 初期化パラメータ（未使用）
        """
        self._spectral_radius = None
        self._previous_diff = None
        self._iteration_count = 0
        self._residual_history = []
    
    @property
    def omega(self) -> float:
        """緩和係数を取得"""
        return self._omega
    
    @omega.setter
    def omega(self, value: float):
        """緩和係数を設定
        
        Args:
            value: 設定する緩和係数
            
        Raises:
            ValueError: 不適切な値が指定された場合
        """
        if value <= 1.0 or value >= 2.0:
            raise ValueError("緩和係数は1と2の間である必要があります")
        self._omega = value
    
    def iterate(self, solution: np.ndarray,
               rhs: np.ndarray,
               dx: float,
               **kwargs) -> np.ndarray:
        """1回のSOR反復を実行
        
        Args:
            solution: 現在の解
            rhs: 右辺
            dx: グリッド間隔
            **kwargs: 未使用
            
        Returns:
            更新された解
        """
        result = solution.copy()
        
        if self.use_redblack:
            # 赤黒順序付けによる更新
            for color in [0, 1]:  # 0: 赤, 1: 黒
                mask = np.zeros_like(result, dtype=bool)
                for i in range(result.ndim):
                    mask ^= np.arange(result.shape[i])[:, None, None] % 2 == color
                
                # 近傍点の和を計算
                neighbors_sum = np.zeros_like(result)
                for axis in range(result.ndim):
                    neighbors_sum += (
                        np.roll(result, 1, axis=axis) +
                        np.roll(result, -1, axis=axis)
                    )
                
                # SOR更新
                gauss_seidel = (
                    dx**2 * rhs[mask] + neighbors_sum[mask]
                ) / (2 * result.ndim)
                result[mask] = (1 - self.omega) * result[mask] + \
                             self.omega * gauss_seidel
                
        else:
            # 通常のSOR反復
            for axis in range(result.ndim):
                neighbors_sum = (
                    np.roll(result, 1, axis=axis) +
                    np.roll(result, -1, axis=axis)
                )
                gauss_seidel = (dx**2 * rhs + neighbors_sum) / (2 * result.ndim)
                result = (1 - self.omega) * result + self.omega * gauss_seidel
        
        # 境界条件の適用
        result = self.apply_boundary_conditions(result)
        
        # 必要に応じて緩和係数を調整
        if self.auto_tune and self.iteration_count % self._update_interval == 0:
            self._update_omega(solution, result)
        
        return result
    
    def _update_omega(self, old_solution: np.ndarray,
                     new_solution: np.ndarray):
        """緩和係数を自動調整
        
        ヤコビ法の反復行列のスペクトル半径を推定し、
        最適な緩和係数を計算します。
        
        Args:
            old_solution: 前回の解
            new_solution: 現在の解
        """
        # 解の変化から反復行列のスペクトル半径を推定
        diff = new_solution - old_solution
        if self.iteration_count > self._update_interval:
            old_diff = self._previous_diff
            numerator = np.sum(diff * diff)
            denominator = np.sum(old_diff * old_diff)
            
            if denominator > 1e-10:
                new_radius = np.sqrt(numerator / denominator)
                
                if self._spectral_radius is None:
                    self._spectral_radius = new_radius
                else:
                    # 指数移動平均で更新
                    alpha = 0.2
                    self._spectral_radius = (
                        (1 - alpha) * self._spectral_radius +
                        alpha * new_radius
                    )
                
                # 最適な緩和係数を計算
                self.omega = 2 / (1 + np.sqrt(1 - self._spectral_radius**2))
        
        # 現在の差分を保存
        self._previous_diff = diff.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """ソルバーの状態を取得"""
        status = super().get_status()
        status.update({
            'method': 'SOR',
            'omega': self.omega,
            'spectral_radius': self._spectral_radius,
            'redblack': self.use_redblack,
            'auto_tune': self.auto_tune
        })
        return status