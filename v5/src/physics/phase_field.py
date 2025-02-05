import numpy as np
from dataclasses import dataclass
from typing import Tuple
from ..core.scheme import DifferenceScheme
from ..core.boundary import DirectionalBC

@dataclass
class PhaseFieldParams:
    epsilon: float = 0.01  # 界面厚さ
    mobility: float = 1.0  # 移動度
    surface_tension: float = 0.07  # 表面張力

class PhaseFieldSolver:
    def __init__(self, 
                 scheme: DifferenceScheme,
                 boundary_conditions: DirectionalBC,
                 params: PhaseFieldParams):
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions
        self.params = params
    
    def initialize_field(self, shape: Tuple[int, ...], 
                        sphere_center: Tuple[float, ...],
                        sphere_radius: float,
                        top_phase_level: float) -> np.ndarray:
        """相場の初期化"""
        phi = np.zeros(shape)
        x, y, z = self._create_grid(shape)
        
        # 球体の設定
        r = np.sqrt((x - sphere_center[0])**2 + 
                   (y - sphere_center[1])**2 + 
                   (z - sphere_center[2])**2)
        phi[r <= sphere_radius] = 1.0
        
        # 上部領域の設定
        phi[..., z[0,0,:] > top_phase_level] = 1.0
        
        return phi
    
    def heaviside(self, phi: np.ndarray) -> np.ndarray:
        """ヘヴィサイド関数の近似"""
        return 0.5 * (1.0 + np.tanh(phi / self.params.epsilon))
    
    def delta(self, phi: np.ndarray) -> np.ndarray:
        """デルタ関数の近似"""
        return (1.0 / (2.0 * self.params.epsilon)) * (
            1.0 - np.tanh(phi / self.params.epsilon)**2
        )
    
    def compute_chemical_potential(self, phi: np.ndarray) -> np.ndarray:
        """化学ポテンシャルの計算"""
        mu = phi * (phi**2 - 1.0) - self.params.epsilon**2 * self.compute_laplacian(phi)
        return mu
    
    def compute_laplacian(self, phi: np.ndarray) -> np.ndarray:
        """ラプラシアンの計算"""
        laplacian = np.zeros_like(phi)
        for axis in range(phi.ndim):
            bc = self.boundary_conditions.get_condition(axis)
            for idx in self._get_orthogonal_indices(phi.shape, axis):
                line = self._get_line(phi, axis, idx)
                d2_line = self.scheme.apply(line, bc)
                self._set_line(laplacian, axis, idx, d2_line)
        return laplacian
    
    def advance(self, phi: np.ndarray, velocity: Tuple[np.ndarray, ...], dt: float) -> np.ndarray:
        """時間発展"""
        # 移流項
        dphi_dt = np.zeros_like(phi)
        for axis, v in enumerate(velocity):
            bc = self.boundary_conditions.get_condition(axis)
            dphi_dt -= v * self.compute_gradient(phi, axis)
        
        # 拡散項
        mu = self.compute_chemical_potential(phi)
        dphi_dt += self.params.mobility * self.compute_laplacian(mu)
        
        return phi + dt * dphi_dt
    
    def _create_grid(self, shape: Tuple[int, ...]) -> Tuple[np.ndarray, ...]:
        """計算グリッドの生成"""
        return np.meshgrid(*[np.linspace(0, 1, n) for n in shape], indexing='ij')
    
    def _get_orthogonal_indices(self, shape: Tuple[int, ...], axis: int):
        ranges = [range(s) for i, s in enumerate(shape) if i != axis]
        return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(len(ranges), -1).T
    
    def _get_line(self, array: np.ndarray, axis: int, idx) -> np.ndarray:
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        return array[tuple(idx_list)]
    
    def _set_line(self, array: np.ndarray, axis: int, idx, values: np.ndarray):
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        array[tuple(idx_list)] = values
    
    def compute_gradient(self, phi: np.ndarray, axis: int) -> np.ndarray:
        """指定方向の勾配を計算"""
        bc = self.boundary_conditions.get_condition(axis)
        gradient = np.zeros_like(phi)
        
        for idx in self._get_orthogonal_indices(phi.shape, axis):
            line = self._get_line(phi, axis, idx)
            grad_line = self.scheme.apply(line, bc)
            self._set_line(gradient, axis, idx, grad_line)
        
        return gradient