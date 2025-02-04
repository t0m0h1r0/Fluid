from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import numba

@dataclass
class PhysicalParameters:
    """物理パラメータの定義"""
    gravity: float = 9.81  # 重力加速度 [m/s²]
    rho_water: float = 1000.0  # 水の密度 [kg/m³]
    rho_nitrogen: float = 1.225  # 窒素の密度 [kg/m³]
    viscosity: float = 1.0e-6  # 動粘性係数 [m²/s]
    surface_tension: float = 0.072  # 表面張力係数 [N/m]

class PhysicsModel:
    """二層流体の物理モデル"""
    def __init__(self, params: PhysicalParameters):
        self.params = params

    @staticmethod
    @numba.njit(parallel=True)
    def compute_density_distribution(
        volume_fractions: np.ndarray, 
        params: PhysicalParameters
    ) -> np.ndarray:
        """密度分布の計算"""
        density = np.zeros_like(volume_fractions, dtype=np.float64)
        
        for i in numba.prange(volume_fractions.shape[0]):
            for j in range(volume_fractions.shape[1]):
                for k in range(volume_fractions.shape[2]):
                    # 体積分率に基づく密度補間
                    density[i,j,k] = (
                        params.rho_water * (1 - volume_fractions[i,j,k]) +
                        params.rho_nitrogen * volume_fractions[i,j,k]
                    )
        
        return density

    @staticmethod
    @numba.njit
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

    @staticmethod
    @numba.njit
    def smooth_heaviside(
        x: np.ndarray, 
        epsilon: float = 1e-3
    ) -> np.ndarray:
        """スムーズ化されたヘビサイド関数"""
        return 0.5 * (1 + np.tanh(x/epsilon))

    def generate_initial_conditions(
        self, 
        grid_x: np.ndarray, 
        grid_y: np.ndarray, 
        grid_z: np.ndarray,
        bubble_center: Tuple[float, float, float] = (0.5, 0.5, 0.2),
        bubble_radius: float = 0.1,
        interface_height: float = 1.8
    ) -> Dict[str, np.ndarray]:
        """初期条件の生成"""
        # メッシュグリッドの生成
        X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        
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
        density = self.compute_density_distribution(volume_fraction, self.params)
        
        # 初期速度場（静止）
        velocity_u = np.zeros_like(X)
        velocity_v = np.zeros_like(X)
        velocity_w = np.zeros_like(X)
        
        # 静水圧分布
        pressure = self.params.rho_water * self.params.gravity * (grid_z.max() - Z)
        
        return {
            'density': density,
            'velocity_u': velocity_u,
            'velocity_v': velocity_v,
            'velocity_w': velocity_w,
            'pressure': pressure,
            'volume_fraction': volume_fraction
        }
