from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class FluidProperties:
    """流体の物性値"""
    density: float
    viscosity: float
    surface_tension: Optional[float] = None
    specific_heat: Optional[float] = None
    thermal_conductivity: Optional[float] = None

class MaterialProperty:
    """物性値の基底クラス"""
    def __init__(self, field_data: np.ndarray):
        self.data = field_data

    def get_value(self, i: int, j: int, k: int) -> float:
        return self.data[i, j, k]

    def get_gradient(self, dx: float) -> np.ndarray:
        return np.gradient(self.data, dx)

class PhaseField:
    """相場"""
    def __init__(self, grid_shape: tuple):
        self.data = np.zeros(grid_shape)
        self.epsilon = 0.01  # 界面厚さパラメータ

    def get_heaviside(self) -> np.ndarray:
        """ヘビサイド関数"""
        return 0.5 * (1 + np.tanh(self.data / self.epsilon))

    def get_delta(self) -> np.ndarray:
        """デルタ関数"""
        return (1.0 / (2.0 * self.epsilon)) * (1 - np.tanh(self.data / self.epsilon)**2)

class MaterialManager:
    """物性値管理クラス"""
    def __init__(self):
        self.fluids: Dict[str, FluidProperties] = {}
        self.phase_field: Optional[PhaseField] = None

    def add_fluid(self, name: str, properties: FluidProperties):
        self.fluids[name] = properties

    def initialize_phase_field(self, grid_shape: tuple):
        self.phase_field = PhaseField(grid_shape)

    def compute_mixture_density(self) -> np.ndarray:
        """混合密度の計算"""
        if not self.phase_field:
            raise ValueError("Phase field has not been initialized")
        
        H = self.phase_field.get_heaviside()
        density = np.zeros_like(H)
        
        for fluid in self.fluids.values():
            density += H * fluid.density
            
        return density

    def compute_mixture_viscosity(self) -> np.ndarray:
        """混合粘性係数の計算"""
        if not self.phase_field:
            raise ValueError("Phase field has not been initialized")
        
        H = self.phase_field.get_heaviside()
        viscosity = np.zeros_like(H)
        
        for fluid in self.fluids.values():
            viscosity += H * fluid.viscosity
            
        return viscosity

    def compute_surface_tension(self) -> np.ndarray:
        """表面張力の計算"""
        if not self.phase_field:
            raise ValueError("Phase field has not been initialized")
        
        # 界面の法線ベクトルを計算
        grad_phi = np.gradient(self.phase_field.data)
        grad_norm = np.sqrt(sum(g**2 for g in grad_phi))
        grad_norm = np.maximum(grad_norm, 1e-10)
        
        # 界面の曲率を計算
        kappa = np.zeros_like(self.phase_field.data)
        for i in range(3):
            grad_norm_i = np.gradient(grad_phi[i] / grad_norm)
            kappa += grad_norm_i[i]
        
        # 表面張力を計算
        surface_tension = np.zeros_like(kappa)
        delta = self.phase_field.get_delta()
        
        # 各流体の表面張力係数を考慮
        for fluid in self.fluids.values():
            if fluid.surface_tension is not None:
                surface_tension += fluid.surface_tension * kappa * delta
        
        return surface_tension

    def validate_properties(self) -> bool:
        """物性値の妥当性検証"""
        if not self.fluids:
            return False
            
        for fluid in self.fluids.values():
            if fluid.density <= 0 or fluid.viscosity <= 0:
                return False
            if fluid.surface_tension is not None and fluid.surface_tension < 0:
                return False
            if fluid.specific_heat is not None and fluid.specific_heat <= 0:
                return False
            if fluid.thermal_conductivity is not None and fluid.thermal_conductivity <= 0:
                return False
                
        return True