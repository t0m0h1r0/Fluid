import numpy as np
from numba import njit
from typing import Dict, Tuple

class PhysicalParameters:
    """物理パラメータの定義"""
    def __init__(
        self, 
        gravity: float = 9.81,
        rho_water: float = 1000.0,
        rho_nitrogen: float = 1.225,
        viscosity: float = 1.0e-6,
        surface_tension: float = 0.072
    ):
        self.gravity = gravity
        self.rho_water = rho_water
        self.rho_nitrogen = rho_nitrogen
        self.viscosity = viscosity
        self.surface_tension = surface_tension

class PhysicsModel:
    def __init__(self, params: PhysicalParameters):
        self.params = params

    @staticmethod
    @njit
    def smooth_heaviside(x: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
        """スムーズ化されたヘビサイド関数"""
        return 0.5 * (1 + np.tanh(x/epsilon))

    @staticmethod
    @njit
    def compute_bubble_distance(
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray, 
        bubble_center: Tuple[float, float, float],
        bubble_radius: float
    ) -> np.ndarray:
        """気泡までの距離計算"""
        return np.sqrt(
            (x - bubble_center[0])**2 + 
            (y - bubble_center[1])**2 + 
            (z - bubble_center[2])**2
        ) - bubble_radius

    def generate_initial_conditions(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        z: np.ndarray,
        bubble_center: Tuple[float, float, float] = (0.5, 0.5, 0.2),
        bubble_radius: float = 0.1,
        interface_height: float = 1.8
    ) -> Dict[str, np.ndarray]:
        """初期条件の生成"""
        # メッシュグリッドの生成
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 距離関数の計算
        bubble_distance = self.compute_bubble_distance(
            X, Y, Z, bubble_center, bubble_radius
        )
        interface_distance = Z - interface_height
        
        # 体積分率の計算
        volume_fraction = (
            self.smooth_heaviside(interface_distance) *
            (1 - self.smooth_heaviside(bubble_distance))
        )
        
        # 密度分布の計算
        density = (
            self.params.rho_water * (1 - volume_fraction) +
            self.params.rho_nitrogen * volume_fraction
        )
        
        # 初期速度場（静止）
        velocity_u = np.zeros_like(X)
        velocity_v = np.zeros_like(X)
        velocity_w = np.zeros_like(X)
        
        # 静水圧分布
        pressure = self.params.rho_water * self.params.gravity * (z.max() - Z)
        
        return {
            'density': density,
            'velocity_u': velocity_u,
            'velocity_v': velocity_v,
            'velocity_w': velocity_w,
            'pressure': pressure,
            'volume_fraction': volume_fraction
        }