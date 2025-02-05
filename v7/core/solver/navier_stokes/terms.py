from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict
from ...field.scalar_field import ScalarField
from ...field.vector_field import VectorField

class NavierStokesTerm(ABC):
    """Navier-Stokes方程式の各項の基底クラス"""
    @abstractmethod
    def compute(self, fields: Dict[str, ScalarField | VectorField]) -> List[np.ndarray]:
        """項の計算"""
        pass

class AdvectionTerm(NavierStokesTerm):
    """移流項: (u・∇)u"""
    def compute(self, fields: Dict[str, ScalarField | VectorField]) -> List[np.ndarray]:
        velocity = fields['velocity'].data
        dx = fields['velocity'].dx
        result = [np.zeros_like(v) for v in velocity]
        
        for i in range(3):  # 各成分について
            for j in range(3):  # 各方向の微分
                result[i] += velocity[j] * np.gradient(velocity[i], dx[j], axis=j)
        
        return result

class PressureTerm(NavierStokesTerm):
    """圧力項: -∇p/ρ"""
    def compute(self, fields: Dict[str, ScalarField | VectorField]) -> List[np.ndarray]:
        pressure = fields['pressure'].data
        density = fields['density'].data
        dx = fields['pressure'].dx
        result = []
        
        for i in range(3):
            grad_p = np.gradient(pressure, dx[i], axis=i)
            result.append(-grad_p / density)
        
        return result

class ViscousTerm(NavierStokesTerm):
    """粘性項: ν∇²u"""
    def compute(self, fields: Dict[str, ScalarField | VectorField]) -> List[np.ndarray]:
        velocity = fields['velocity'].data
        viscosity = fields['viscosity'].data
        dx = fields['velocity'].dx
        result = []
        
        for i in range(3):
            laplacian = np.zeros_like(velocity[i])
            for j in range(3):
                laplacian += np.gradient(
                    viscosity * np.gradient(velocity[i], dx[j], axis=j),
                    dx[j], axis=j
                )
            result.append(laplacian)
        
        return result

class ExternalForceTerm(NavierStokesTerm):
    """外力項: F"""
    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity

    def compute(self, fields: Dict[str, ScalarField | VectorField]) -> List[np.ndarray]:
        velocity = fields['velocity'].data
        density = fields['density'].data
        
        # 重力項 (z方向のみ)
        force = [
            np.zeros_like(velocity[0]),  # x方向
            np.zeros_like(velocity[1]),  # y方向
            -self.gravity * density      # z方向
        ]
        
        return force