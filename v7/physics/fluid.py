import numpy as np
from dataclasses import dataclass
from typing import Optional
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

@dataclass
class FluidProperties:
    """流体の物性値"""
    density: float
    viscosity: float
    specific_heat: Optional[float] = None
    thermal_conductivity: Optional[float] = None

class Fluid:
    """流体クラス"""
    
    def __init__(self, 
                 properties: FluidProperties,
                 density_field: ScalarField,
                 velocity_field: VectorField,
                 pressure_field: ScalarField):
        self.properties = properties
        self.density = density_field
        self.velocity = velocity_field
        self.pressure = pressure_field
        self._validate_fields()

    def _validate_fields(self):
        """フィールドの整合性チェック"""
        shapes = [
            self.density.data.shape,
            self.pressure.data.shape
        ] + [v.shape for v in self.velocity.data]
        
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("全てのフィールドは同じ形状である必要があります")

    def get_kinetic_energy(self) -> np.ndarray:
        """運動エネルギーの計算"""
        return 0.5 * self.density.data * sum(
            v**2 for v in self.velocity.data
        )

    def get_momentum(self) -> list[np.ndarray]:
        """運動量の計算"""
        return [
            self.density.data * v 
            for v in self.velocity.data
        ]

    def get_vorticity(self) -> list[np.ndarray]:
        """渦度の計算"""
        dx = self.velocity.dx
        u, v, w = self.velocity.data
        
        # 渦度 ω = ∇ × u
        dudy = np.gradient(u, dx[1], axis=1)
        dudz = np.gradient(u, dx[2], axis=2)
        dvdx = np.gradient(v, dx[0], axis=0)
        dvdz = np.gradient(v, dx[2], axis=2)
        dwdx = np.gradient(w, dx[0], axis=0)
        dwdy = np.gradient(w, dx[1], axis=1)
        
        return [
            dwdy - dvdz,  # ωx
            dudz - dwdx,  # ωy
            dvdx - dudy   # ωz
        ]

    def get_strain_rate(self) -> list[list[np.ndarray]]:
        """ひずみ速度テンソルの計算"""
        dx = self.velocity.dx
        u, v, w = self.velocity.data
        
        # ひずみ速度テンソル Sij = (∂ui/∂xj + ∂uj/∂xi)/2
        strain = [[None for _ in range(3)] for _ in range(3)]
        
        for i, ui in enumerate([u, v, w]):
            for j, uj in enumerate([u, v, w]):
                duidxj = np.gradient(ui, dx[j], axis=j)
                dujdxi = np.gradient(uj, dx[i], axis=i)
                strain[i][j] = 0.5 * (duidxj + dujdxi)
        
        return strain

    def get_viscous_stress(self) -> list[list[np.ndarray]]:
        """粘性応力テンソルの計算"""
        strain = self.get_strain_rate()
        
        # 粘性応力テンソル τij = 2μSij
        stress = [[None for _ in range(3)] for _ in range(3)]
        
        for i in range(3):
            for j in range(3):
                stress[i][j] = 2 * self.properties.viscosity * strain[i][j]
        
        return stress