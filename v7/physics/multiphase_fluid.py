import numpy as np
from typing import Dict, List, Optional
from .fluid import Fluid, FluidProperties
from .phase_field import PhaseField, PhaseFieldParameters
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class MultiPhaseFluid:
    """混相流体クラス"""
    def __init__(self, 
                 fluids: Dict[str, Fluid],
                 phase_field: PhaseField):
        self.fluids = fluids
        self.phase_field = phase_field
        self._validate_fields()
        self._initialize_mixture_properties()

    def _validate_fields(self):
        """フィールドの整合性チェック"""
        shape = self.phase_field.phi.data.shape
        for fluid in self.fluids.values():
            if fluid.density.data.shape != shape:
                raise ValueError("全てのフィールドは同じ形状である必要があります")

    def _initialize_mixture_properties(self):
        """混合物性の初期化"""
        self.mixture_density = ScalarField(self.phase_field.phi.metadata)
        self.mixture_viscosity = ScalarField(self.phase_field.phi.metadata)
        self._update_mixture_properties()

    def _update_mixture_properties(self):
        """混合物性の更新"""
        H = self.phase_field.compute_heaviside_function()
        
        # 密度の混合則
        density = np.zeros_like(H)
        for fluid in self.fluids.values():
            density += H * fluid.properties.density
        self.mixture_density.data = density

        # 粘性の混合則（調和平均）
        viscosity = np.zeros_like(H)
        for fluid in self.fluids.values():
            viscosity += H / fluid.properties.viscosity
        self.mixture_viscosity.data = 1.0 / viscosity

    def evolve(self, dt: float):
        """時間発展"""
        # Phase-Fieldの更新
        velocity = next(iter(self.fluids.values())).velocity
        self.phase_field.evolve(velocity, dt)
        
        # 混合物性の更新
        self._update_mixture_properties()
        
        # 界面力の計算
        surface_tension = self.phase_field.compute_surface_tension_force()
        
        # 各流体の更新
        for fluid in self.fluids.values():
            # 密度・粘性の更新
            H = self.phase_field.compute_heaviside_function()
            fluid.density.data = self.mixture_density.data
            
            # 速度・圧力の更新は外部のソルバーで行う

    def get_kinetic_energy(self) -> float:
        """系全体の運動エネルギー"""
        energy = 0.0
        for fluid in self.fluids.values():
            energy += np.sum(fluid.get_kinetic_energy())
        return energy

    def get_surface_energy(self) -> float:
        """界面エネルギー"""
        return self.phase_field.compute_mixing_energy()

    def get_total_mass(self) -> float:
        """系全体の質量"""
        return np.sum(self.mixture_density.data)

    def get_center_of_mass(self) -> np.ndarray:
        """質量中心の計算"""
        dx = self.mixture_density.dx
        x = np.arange(0, self.mixture_density.data.shape[0]) * dx[0]
        y = np.arange(0, self.mixture_density.data.shape[1]) * dx[1]
        z = np.arange(0, self.mixture_density.data.shape[2]) * dx[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        mass = self.get_total_mass()
        com_x = np.sum(X * self.mixture_density.data) / mass
        com_y = np.sum(Y * self.mixture_density.data) / mass
        com_z = np.sum(Z * self.mixture_density.data) / mass
        
        return np.array([com_x, com_y, com_z])

    def get_total_momentum(self) -> np.ndarray:
        """系全体の運動量"""
        momentum = np.zeros(3)
        for fluid in self.fluids.values():
            for i, p in enumerate(fluid.get_momentum()):
                momentum[i] += np.sum(p)
        return momentum

    def get_interface_statistics(self) -> Dict[str, float]:
        """界面の統計情報"""
        # 界面位置の取得
        x, y, z = self.phase_field.get_interface_location()
        if len(x) == 0:
            return {
                'area': 0.0,
                'mean_curvature': 0.0,
                'max_curvature': 0.0
            }
        
        # 界面の法線と曲率
        curvature = self.phase_field.compute_curvature()
        
        # 統計量の計算
        dx = self.phase_field.phi.dx
        cell_area = dx[0] * dx[1]  # 簡略化された面積計算
        
        interface_area = len(x) * cell_area
        mean_curvature = np.mean(np.abs(curvature))
        max_curvature = np.max(np.abs(curvature))
        
        return {
            'area': interface_area,
            'mean_curvature': mean_curvature,
            'max_curvature': max_curvature
        }