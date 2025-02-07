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

    def _safe_gradient(self, data: np.ndarray, dx: float, axis: int) -> np.ndarray:
        """安全な勾配計算（境界を適切に処理）"""
        grad = np.zeros_like(data)
        
        # 内部領域（中心差分）
        slices_inner = [slice(1, -1) if i == axis else slice(None) 
                       for i in range(data.ndim)]
        grad[tuple(slices_inner)] = (
            data[tuple(slice(2, None) if i == axis else slice(None) 
                      for i in range(data.ndim))] -
            data[tuple(slice(0, -2) if i == axis else slice(None) 
                      for i in range(data.ndim))]
        ) / (2.0 * dx)
        
        # 境界（片側差分）
        # 左境界
        slices_left = [slice(0, 1) if i == axis else slice(None) 
                      for i in range(data.ndim)]
        grad[tuple(slices_left)] = (
            data[tuple(slice(1, 2) if i == axis else slice(None) 
                      for i in range(data.ndim))] -
            data[tuple(slices_left)]
        ) / dx
        
        # 右境界
        slices_right = [slice(-1, None) if i == axis else slice(None) 
                       for i in range(data.ndim)]
        grad[tuple(slices_right)] = (
            data[tuple(slices_right)] -
            data[tuple(slice(-2, -1) if i == axis else slice(None) 
                      for i in range(data.ndim))]
        ) / dx
        
        return grad

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
            return self._compute_weno_safe(velocity)
        else:
            return self._compute_standard_safe(velocity)
    
    def _compute_standard_safe(self, velocity: VectorField) -> List[np.ndarray]:
        """標準的な中心差分による移流項の計算（安全版）"""
        result = []
        dx = velocity.dx
        
        for i in range(len(velocity.components)):
            conv = np.zeros_like(velocity.components[i].data)
            
            for j in range(len(velocity.components)):
                grad = self._safe_gradient(velocity.components[i].data, dx, j)
                conv -= velocity.components[j].data * grad
            
            result.append(conv)
        
        return result
    
    def _compute_weno_safe(self, velocity: VectorField) -> List[np.ndarray]:
        """WENO法による移流項の計算（安全版）"""
        result = []
        dx = velocity.dx
        
        for i in range(len(velocity.components)):
            conv = np.zeros_like(velocity.components[i].data)
            
            for j in range(len(velocity.components)):
                # 内部領域のみでWENOを使用
                for k in range(2, velocity.shape[0]-2):
                    for l in range(2, velocity.shape[1]-2):
                        for m in range(2, velocity.shape[2]-2):
                            v = velocity.components[j].data[k,l,m]
                            if abs(v) < 1e-10:
                                continue
                                
                            # 方向に応じたステンシルの選択
                            if j == 0:
                                stencil = velocity.components[i].data[k-2:k+3,l,m]
                            elif j == 1:
                                stencil = velocity.components[i].data[k,l-2:l+3,m]
                            else:  # j == 2
                                stencil = velocity.components[i].data[k,l,m-2:m+3]
                            
                            # WENO5による勾配計算
                            conv[k,l,m] -= v * self._weno5_flux(stencil, v > 0) / dx
            
            result.append(conv)
        return result
    
    def _weno5_flux(self, stencil: np.ndarray, is_positive: bool) -> float:
        """WENO5法によるフラックスの計算"""
        eps = 1e-6
        
        if is_positive:
            v1, v2, v3, v4, v5 = stencil
        else:
            v1, v2, v3, v4, v5 = stencil[::-1]
        
        # 3つの候補ステンシル
        s1 = (13/12) * (v1 - 2*v2 + v3)**2 + (1/4) * (v1 - 4*v2 + 3*v3)**2
        s2 = (13/12) * (v2 - 2*v3 + v4)**2 + (1/4) * (v2 - v4)**2
        s3 = (13/12) * (v3 - 2*v4 + v5)**2 + (1/4) * (3*v3 - 4*v4 + v5)**2
        
        # 非線形重み
        alpha1 = 0.1 / (eps + s1)**2
        alpha2 = 0.6 / (eps + s2)**2
        alpha3 = 0.3 / (eps + s3)**2
        
        w1 = alpha1 / (alpha1 + alpha2 + alpha3)
        w2 = alpha2 / (alpha1 + alpha2 + alpha3)
        w3 = alpha3 / (alpha1 + alpha2 + alpha3)
        
        # フラックスの構築
        q1 = (1/6) * (2*v1 - 7*v2 + 11*v3)
        q2 = (1/6) * (-v2 + 5*v3 + 2*v4)
        q3 = (1/6) * (2*v3 + 5*v4 - v5)
        
        return w1*q1 + w2*q2 + w3*q3

class DiffusionTerm(Term):
    """粘性項"""
    
    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        """粘性項の計算（境界を適切に処理）"""
        viscosity = kwargs.get('viscosity', None)
        density = kwargs.get('density', None)
        
        if viscosity is None or density is None:
            raise ValueError("粘性係数と密度が必要です")
        
        result = []
        dx = velocity.dx
        
        for i in range(len(velocity.components)):
            # ラプラシアンの計算
            diff = np.zeros_like(velocity.components[i].data)
            for j in range(velocity.components[i].data.ndim):
                grad = self._safe_gradient(velocity.components[i].data, dx, j)
                diff += self._safe_gradient(grad, dx, j)
            
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
            grad_p = self._safe_gradient(pressure, velocity.dx, i)
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
        
        # 結果配列の初期化
        result = [np.zeros_like(comp.data) for comp in velocity.components]
        
        # 重力の追加（z方向）
        result[-1] = -self.gravity * np.ones_like(velocity.components[-1].data)
        
        # 表面張力の追加（指定されている場合）
        if surface_tension is not None:
            for i in range(len(result)):
                result[i] += surface_tension[i] / density
        
        return result