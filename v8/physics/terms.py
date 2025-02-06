from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional
from core.field import Field, VectorField
from dataclasses import dataclass

class Term(ABC):
    """ナビエ・ストークス方程式の項の基底クラス"""
    
    @abstractmethod
    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        """項の計算
        
        Args:
            velocity: 速度場
            **kwargs: 追加のパラメータ
            
        Returns:
            各方向の項の値
        """
        pass

class ConvectionTerm(Term):
    """移流項"""
    
    def __init__(self, use_weno: bool = True):
        """
        Args:
            use_weno: WENO法を使用するかどうか
        """
        self.use_weno = use_weno
    
    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        if self.use_weno:
            return self._compute_weno(velocity)
        else:
            return self._compute_standard(velocity)
    
    def _compute_standard(self, velocity: VectorField) -> List[np.ndarray]:
        """標準的な中心差分による移流項の計算"""
        result = []
        for i in range(len(velocity.components)):
            conv = np.zeros_like(velocity.components[i].data)
            for j in range(len(velocity.components)):
                conv += velocity.components[j].data * \
                       velocity.components[i].gradient(j)
            result.append(-conv)
        return result
    
    def _compute_weno(self, velocity: VectorField) -> List[np.ndarray]:
        """WENO法による移流項の計算"""
        result = []
        for i in range(len(velocity.components)):
            conv = np.zeros_like(velocity.components[i].data)
            for j in range(len(velocity.components)):
                v = velocity.components[j].data
                # 風上差分の方向を決定
                upwind = v < 0
                
                # WENO5による風上差分
                flux_plus = self._weno5(velocity.components[i].data, j, False)
                flux_minus = self._weno5(velocity.components[i].data, j, True)
                
                # フラックスの合成
                flux = np.where(upwind, flux_minus, flux_plus)
                conv += v * flux
            result.append(-conv)
        return result
    
    def _weno5(self, field: np.ndarray, axis: int, is_negative: bool) -> np.ndarray:
        """5次精度WENOスキーム"""
        # WENOの重み係数
        epsilon = 1e-6
        gamma0, gamma1, gamma2 = 0.1, 0.6, 0.3
        
        # シフトしたインデックスを準備
        if is_negative:
            v1 = np.roll(field, 2, axis=axis)
            v2 = np.roll(field, 1, axis=axis)
            v3 = field
            v4 = np.roll(field, -1, axis=axis)
            v5 = np.roll(field, -2, axis=axis)
        else:
            v1 = np.roll(field, -2, axis=axis)
            v2 = np.roll(field, -1, axis=axis)
            v3 = field
            v4 = np.roll(field, 1, axis=axis)
            v5 = np.roll(field, 2, axis=axis)
        
        # 3つの候補ステンシル
        s0 = (13/12) * (v1 - 2*v2 + v3)**2 + (1/4) * (v1 - 4*v2 + 3*v3)**2
        s1 = (13/12) * (v2 - 2*v3 + v4)**2 + (1/4) * (v2 - v4)**2
        s2 = (13/12) * (v3 - 2*v4 + v5)**2 + (1/4) * (3*v3 - 4*v4 + v5)**2
        
        # 非線形重み
        alpha0 = gamma0 / (epsilon + s0)**2
        alpha1 = gamma1 / (epsilon + s1)**2
        alpha2 = gamma2 / (epsilon + s2)**2
        omega = np.array([alpha0, alpha1, alpha2])
        omega /= np.sum(omega, axis=0)
        
        # 各ステンシルでの補間値
        p0 = (1/6) * (2*v1 - 7*v2 + 11*v3)
        p1 = (1/6) * (-v2 + 5*v3 + 2*v4)
        p2 = (1/6) * (2*v3 + 5*v4 - v5)
        
        return omega[0]*p0 + omega[1]*p1 + omega[2]*p2

class DiffusionTerm(Term):
    """粘性項"""
    
    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        viscosity = kwargs.get('viscosity', None)
        density = kwargs.get('density', None)
        
        if viscosity is None or density is None:
            raise ValueError("粘性係数と密度が必要です")
            
        result = []
        for i in range(len(velocity.components)):
            # ラプラシアンの計算
            diff = velocity.components[i].laplacian()
            
            # 粘性係数の勾配による追加項
            grad_mu = [np.gradient(viscosity, axis=j) for j in range(velocity.ndim)]
            grad_u = [velocity.components[i].gradient(j) for j in range(velocity.ndim)]
            
            # ∇・(μ(∇u + ∇u^T))の計算
            diff += sum(grad_mu[j] * grad_u[j] for j in range(velocity.ndim))
            
            # 密度で割って加速度に変換
            result.append(viscosity * diff / density)
            
        return result

class PressureTerm(Term):
    """圧力項"""
    
    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        pressure = kwargs.get('pressure', None)
        density = kwargs.get('density', None)
        
        if pressure is None or density is None:
            raise ValueError("圧力場と密度が必要です")
        
        result = []
        for i in range(len(velocity.components)):
            # 圧力勾配の計算
            grad_p = np.gradient(pressure, axis=i)
            # 密度で割って加速度に変換
            result.append(-grad_p / density)
            
        return result

class ExternalForcesTerm(Term):
    """外力項"""
    
    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity
    
    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        density = kwargs.get('density', None)
        surface_tension = kwargs.get('surface_tension', None)
        
        if density is None:
            raise ValueError("密度が必要です")
            
        # 重力
        result = [np.zeros_like(velocity.components[i].data) 
                 for i in range(len(velocity.components))]
        result[-1] = -self.gravity * np.ones_like(velocity.components[-1].data)
        
        # 表面張力
        if surface_tension is not None:
            for i in range(len(result)):
                result[i] += surface_tension[i]
        
        return result
