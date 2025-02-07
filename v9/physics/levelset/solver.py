"""Level Setソルバーを提供するモジュール

このモジュールは、Level Set方程式を解くためのソルバーを定義します。
WENOスキームを用いた高次精度の解法を実装します。
"""

from typing import Dict, Any
import numpy as np
from core.solver import TemporalSolver
from core.field import VectorField
from .field import LevelSetField

class LevelSetSolver(TemporalSolver):
    """Level Setソルバークラス
    
    このクラスは、Level Set方程式を時間発展させるソルバーを実装します。
    移流方程式に対してWENOスキームを使用し、高次精度で数値振動の少ない解法を提供します。
    """
    
    def __init__(self, 
                 use_weno: bool = True,
                 weno_order: int = 5,
                 **kwargs):
        """Level Setソルバーを初期化
        
        Args:
            use_weno: WENOスキームを使用するかどうか
            weno_order: WENOスキームの次数（3または5）
            **kwargs: 基底クラスに渡すパラメータ
        """
        super().__init__(name="LevelSet", **kwargs)
        self.use_weno = use_weno
        self.weno_order = weno_order
        
        # WENOスキームの係数を初期化
        if use_weno:
            self._init_weno_coefficients()
    
    def initialize(self, **kwargs) -> None:
        """ソルバーの初期化
        
        Args:
            **kwargs: 初期化に必要なパラメータ
        """
        # Level Setソルバーの初期化処理
        # 現時点では特別な初期化は必要ないため、パスします
        pass
    
    def _init_weno_coefficients(self):
        """WENOスキームの係数を初期化"""
        # WENO5の場合の係数
        if self.weno_order == 5:
            # 線形重み
            self.weno_weights = np.array([0.1, 0.6, 0.3])
            
            # 各ステンシルでの係数
            self.weno_coeffs = [
                np.array([1/3, -7/6, 11/6]),      # 左側ステンシル
                np.array([-1/6, 5/6, 1/3]),       # 中央ステンシル
                np.array([1/3, 5/6, -1/6])        # 右側ステンシル
            ]
            
        # WENO3の場合の係数
        elif self.weno_order == 3:
            self.weno_weights = np.array([1/3, 2/3])
            self.weno_coeffs = [
                np.array([-1/2, 3/2]),            # 左側ステンシル
                np.array([1/2, 1/2])              # 右側ステンシル
            ]
        else:
            raise ValueError(f"未対応のWENO次数です: {self.weno_order}")
    
    def _weno_reconstruction(self, values: np.ndarray, axis: int) -> np.ndarray:
        """WENOスキームによる再構築
        
        Args:
            values: 再構築する値の配列
            axis: 再構築を行う軸
            
        Returns:
            再構築された値
        """
        # シフトしたインデックスでの値を取得
        if self.weno_order == 5:
            v1 = np.roll(values, 2, axis=axis)
            v2 = np.roll(values, 1, axis=axis)
            v3 = values
            v4 = np.roll(values, -1, axis=axis)
            v5 = np.roll(values, -2, axis=axis)
            
            # 各ステンシルでの滑らかさ指標を計算
            eps = 1e-6  # ゼロ除算防止用
            beta0 = 13/12 * (v1 - 2*v2 + v3)**2 + 1/4 * (v1 - 4*v2 + 3*v3)**2
            beta1 = 13/12 * (v2 - 2*v3 + v4)**2 + 1/4 * (v2 - v4)**2
            beta2 = 13/12 * (v3 - 2*v4 + v5)**2 + 1/4 * (3*v3 - 4*v4 + v5)**2
            
            # 非線形重みを計算
            alpha = self.weno_weights / (eps + beta0)**2
            omega = alpha / np.sum(alpha, axis=0)
            
            # 各ステンシルでの補間値を計算
            p0 = self.weno_coeffs[0][0]*v1 + self.weno_coeffs[0][1]*v2 + self.weno_coeffs[0][2]*v3
            p1 = self.weno_coeffs[1][0]*v2 + self.weno_coeffs[1][1]*v3 + self.weno_coeffs[1][2]*v4
            p2 = self.weno_coeffs[2][0]*v3 + self.weno_coeffs[2][1]*v4 + self.weno_coeffs[2][2]*v5
            
            return omega[0]*p0 + omega[1]*p1 + omega[2]*p2
            
        else:  # WENO3
            v1 = np.roll(values, 1, axis=axis)
            v2 = values
            v3 = np.roll(values, -1, axis=axis)
            
            beta0 = (v2 - v1)**2
            beta1 = (v3 - v2)**2
            
            eps = 1e-6
            alpha = self.weno_weights / (eps + beta0)**2
            omega = alpha / np.sum(alpha, axis=0)
            
            p0 = self.weno_coeffs[0][0]*v1 + self.weno_coeffs[0][1]*v2
            p1 = self.weno_coeffs[1][0]*v2 + self.weno_coeffs[1][1]*v3
            
            return omega[0]*p0 + omega[1]*p1
    
    def compute_timestep(self, phi: LevelSetField, 
                        velocity: VectorField, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算
        
        Args:
            phi: Level Set場
            velocity: 速度場
            **kwargs: 追加のパラメータ
            
        Returns:
            計算された時間刻み幅
        """
        # 最大速度を計算
        max_velocity = max(np.max(np.abs(v.data)) for v in velocity.components)
        
        # CFL条件に基づく時間刻み幅
        dt = self.cfl * phi.dx / (max_velocity + 1e-10)
        
        return np.clip(dt, self._min_dt, self._max_dt)
    
    def advance(self, dt: float, phi: LevelSetField,
               velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める
        
        Args:
            dt: 時間刻み幅
            phi: Level Set場
            velocity: 速度場
            **kwargs: 追加のパラメータ
            
        Returns:
            計算結果と統計情報を含む辞書
        """
        # 移流項の計算
        if self.use_weno:
            # WENOスキームによる空間離散化
            flux = np.zeros_like(phi.data)
            for i, v in enumerate(velocity.components):
                # 風上差分の方向を決定
                upwind = v.data < 0
                
                # 正の速度に対する flux
                phi_plus = self._weno_reconstruction(phi.data, i)
                # 負の速度に対する flux
                phi_minus = self._weno_reconstruction(np.flip(phi.data, i), i)
                phi_minus = np.flip(phi_minus, i)
                
                # 風上方向に応じてfluxを選択
                flux += v.data * np.where(upwind, phi_minus, phi_plus)
        else:
            # 標準的な中心差分
            flux = sum(v.data * phi.gradient(i) 
                      for i, v in enumerate(velocity.components))
        
        # 時間積分（前進Euler法）
        phi.data = phi.data - dt * flux
        
        # 必要に応じて再初期化
        phi.advance_step()
        if phi.need_reinit():
            phi.reinitialize()
        
        # 統計情報の収集
        diagnostics = phi.get_diagnostics()
        diagnostics.update({
            'dt': dt,
            'max_velocity': max(np.max(np.abs(v.data)) 
                              for v in velocity.components)
        })
        
        return {
            'converged': True,
            'residual': np.max(np.abs(flux)) * dt,
            'diagnostics': diagnostics
        }