import numpy as np
from typing import Dict, List, Optional
from .fluid import Fluid
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class BodyForce:
    """物体力の基底クラス"""
    def compute_force(self, fluids: Dict[str, Fluid]) -> List[np.ndarray]:
        """力の計算"""
        pass

class GravitationalForce(BodyForce):
    """重力"""
    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity

    def compute_force(self, fluids: Dict[str, Fluid]) -> List[np.ndarray]:
        # 密度場を取得
        density = next(iter(fluids.values())).density.data
        
        # 重力加速度による力
        return [
            np.zeros_like(density),  # x方向
            np.zeros_like(density),  # y方向
            -self.gravity * density  # z方向
        ]

class BuoyancyForce(BodyForce):
    """浮力"""
    def __init__(self, reference_density: float):
        self.reference_density = reference_density

    def compute_force(self, fluids: Dict[str, Fluid]) -> List[np.ndarray]:
        density = next(iter(fluids.values())).density.data
        
        # アルキメデスの原理による浮力
        buoyancy = (density - self.reference_density) * 9.81
        
        return [
            np.zeros_like(density),  # x方向
            np.zeros_like(density),  # y方向
            buoyancy                # z方向
        ]

class ThermalExpansionForce(BodyForce):
    """熱膨張による浮力"""
    def __init__(self, 
                 reference_temperature: float,
                 thermal_expansion_coefficient: float):
        self.reference_temperature = reference_temperature
        self.beta = thermal_expansion_coefficient

    def compute_force(self, fluids: Dict[str, Fluid]) -> List[np.ndarray]:
        density = next(iter(fluids.values())).density.data
        temperature = fluids[next(iter(fluids))].temperature.data
        
        # ブジネスク近似による浮力
        buoyancy = density * self.beta * (
            temperature - self.reference_temperature
        ) * 9.81
        
        return [
            np.zeros_like(density),
            np.zeros_like(density),
            buoyancy
        ]

class BodyForceManager:
    """物体力の管理クラス"""
    def __init__(self):
        self.forces: List[BodyForce] = []

    def add_force(self, force: BodyForce):
        self.forces.append(force)

    def compute_total_force(self, fluids: Dict[str, Fluid]) -> List[np.ndarray]:
        # 初期化
        shape = next(iter(fluids.values())).density.data.shape
        total_force = [np.zeros(shape) for _ in range(3)]
        
        # 全ての力の合計
        for force in self.forces:
            f = force.compute_force(fluids)
            for i in range(3):
                total_force[i] += f[i]
        
        return total_force