from abc import ABC, abstractmethod
import numpy as np
from core.field.base import Field
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class ConservationLaw(ABC):
    """保存則の基底クラス"""
    
    @abstractmethod
    def compute_flux(self, fields: dict[str, Field]) -> list[np.ndarray]:
        """フラックスの計算"""
        pass

    @abstractmethod
    def compute_source(self, fields: dict[str, Field]) -> np.ndarray:
        """ソース項の計算"""
        pass

class MassConservation(ConservationLaw):
    """質量保存則"""
    
    def compute_flux(self, fields: dict[str, Field]) -> list[np.ndarray]:
        if 'density' not in fields or 'velocity' not in fields:
            raise KeyError("密度場と速度場が必要です")
            
        density = fields['density'].data
        velocity = fields['velocity'].data
        
        # 質量フラックス ρu
        return [density * v for v in velocity]

    def compute_source(self, fields: dict[str, Field]) -> np.ndarray:
        # 質量保存則にソース項はない
        return np.zeros_like(fields['density'].data)

class MomentumConservation(ConservationLaw):
    """運動量保存則"""
    
    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity

    def compute_flux(self, fields: dict[str, Field]) -> list[np.ndarray]:
        if not all(key in fields for key in ['density', 'velocity', 'pressure']):
            raise KeyError("密度場、速度場、圧力場が必要です")
            
        density = fields['density'].data
        velocity = fields['velocity'].data
        pressure = fields['pressure'].data
        
        # 運動量フラックス ρuu + pI
        flux = []
        for i in range(3):
            f = []
            for j in range(3):
                f_ij = density * velocity[i] * velocity[j]
                if i == j:
                    f_ij += pressure
                f.append(f_ij)
            flux.append(sum(f))
        return flux

    def compute_source(self, fields: dict[str, Field]) -> np.ndarray:
        density = fields['density'].data
        
        # 重力項 ρg
        source = np.zeros((3,) + density.shape)
        source[2] = -self.gravity * density
        return source

class EnergyConservation(ConservationLaw):
    """エネルギー保存則"""
    
    def compute_flux(self, fields: dict[str, Field]) -> list[np.ndarray]:
        if not all(key in fields for key in 
                  ['density', 'velocity', 'pressure', 'temperature']):
            raise KeyError("密度場、速度場、圧力場、温度場が必要です")
            
        density = fields['density'].data
        velocity = fields['velocity'].data
        pressure = fields['pressure'].data
        temperature = fields['temperature'].data
        
        # 運動エネルギー
        kinetic = 0.5 * density * sum(v**2 for v in velocity)
        # 内部エネルギー
        internal = density * temperature  # 単純化
        
        # エネルギーフラックス (E + p)u
        total_energy = kinetic + internal
        return [(total_energy + pressure) * v for v in velocity]

    def compute_source(self, fields: dict[str, Field]) -> np.ndarray:
        # 熱伝導などのソース項（現在は簡略化）
        return np.zeros_like(fields['density'].data)