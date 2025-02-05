# physics/phase_field.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from core.scheme import DifferenceScheme
from core.boundary import DirectionalBC

@dataclass
class PhaseFieldParams:
    epsilon: float = 0.01        # 界面厚さ
    mobility: float = 1.0        # 移動度
    surface_tension: float = 0.07 # 表面張力係数
    delta_min: float = 1e-6      # デルタ関数の最小値
    kappa_max: float = 100.0     # 曲率の最大値

@dataclass
class PhaseObject:
    """相場オブジェクトの基底クラス"""
    phase: str  # LayerConfigと合わせて属性名をphaseに変更

@dataclass
class Layer(PhaseObject):
    """レイヤーオブジェクト"""
    z_range: Tuple[float, float]

@dataclass
class Sphere(PhaseObject):
    """球オブジェクト"""
    center: Tuple[float, float, float]
    radius: float

class PhaseFieldSolver:
    def __init__(self, 
                 scheme: DifferenceScheme,
                 boundary_conditions: DirectionalBC,
                 params: PhaseFieldParams):
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions
        self.params = params
        self._grid = None
        self._phase_densities = {}

    def initialize_field(self, shape: Tuple[int, ...], 
                        domain_size: Tuple[float, ...]) -> np.ndarray:
        """相場の初期化
        
        Args:
            shape: グリッドサイズ (Nx, Ny, Nz)
            domain_size: 物理領域サイズ (Lx, Ly, Lz)
            
        Returns:
            初期化された相場配列
        """
        self._grid = self._create_grid(shape, domain_size)
        return np.zeros(shape)

    def set_phase_density(self, phase_name: str, density: float):
        """相の密度を登録
        
        Args:
            phase_name: 相の名前
            density: 密度値
        """
        self._phase_densities[phase_name] = density

    def add_layer(self, field: np.ndarray, layer: Layer) -> np.ndarray:
        """レイヤーを相場に追加
        
        Args:
            field: 現在の相場
            layer: レイヤー設定
            
        Returns:
            更新された相場
        """
        if self._grid is None:
            raise RuntimeError("グリッドが初期化されていません。initialize_fieldを先に呼び出してください。")
        
        X, Y, Z = self._grid
        z_min, z_max = layer.z_range
        mask = (Z >= z_min) & (Z < z_max)
        
        if layer.phase not in self._phase_densities:  # phase_nameをphaseに変更
            raise ValueError(f"相 {layer.phase} の密度が設定されていません")
            
        field[mask] = self._phase_densities[layer.phase]  # phase_nameをphaseに変更
        return field

    def add_sphere(self, field: np.ndarray, sphere: Sphere) -> np.ndarray:
        """球を相場に追加
        
        Args:
            field: 現在の相場
            sphere: 球の設定
            
        Returns:
            更新された相場
        """
        if self._grid is None:
            raise RuntimeError("グリッドが初期化されていません。initialize_fieldを先に呼び出してください。")
            
        X, Y, Z = self._grid
        r = np.sqrt(
            (X - sphere.center[0])**2 + 
            (Y - sphere.center[1])**2 + 
            (Z - sphere.center[2])**2
        )
        mask = r <= sphere.radius
        
        if sphere.phase not in self._phase_densities:  # phase_nameをphaseに変更
            raise ValueError(f"相 {sphere.phase} の密度が設定されていません")
            
        field[mask] = self._phase_densities[sphere.phase]  # phase_nameをphaseに変更
        return field

    def compute_curvature(self, phi: np.ndarray) -> np.ndarray:
        """界面の曲率を計算
        
        Args:
            phi: 相場
            
        Returns:
            曲率場
        """
        # 勾配の計算
        grad_x = self.compute_gradient(phi, 0)
        grad_y = self.compute_gradient(phi, 1)
        grad_z = self.compute_gradient(phi, 2)
        
        # 二階微分の計算
        grad_xx = self.compute_gradient(grad_x, 0)
        grad_yy = self.compute_gradient(grad_y, 1)
        grad_zz = self.compute_gradient(grad_z, 2)
        
        grad_xy = self.compute_gradient(grad_x, 1)
        grad_yz = self.compute_gradient(grad_y, 2)
        grad_zx = self.compute_gradient(grad_z, 0)
        
        # 勾配の大きさ
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        grad_mag = np.maximum(grad_mag, self.params.delta_min)
        
        # 曲率の計算
        kappa = (
            (grad_xx * (grad_y**2 + grad_z**2) + 
             grad_yy * (grad_z**2 + grad_x**2) + 
             grad_zz * (grad_x**2 + grad_y**2) - 
             2 * (grad_xy * grad_x * grad_y + 
                  grad_yz * grad_y * grad_z + 
                  grad_zx * grad_z * grad_x)
            ) / grad_mag**3
        )
        
        # 数値的安定性のための制限
        return np.clip(kappa, -self.params.kappa_max, self.params.kappa_max)

    def compute_surface_tension(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """表面張力による力を計算
        
        Args:
            phi: 相場
            
        Returns:
            表面張力による力場 (fx, fy, fz)
        """
        kappa = self.compute_curvature(phi)
        delta = self.delta(phi)
        
        # 法線方向の計算
        grad_x = self.compute_gradient(phi, 0)
        grad_y = self.compute_gradient(phi, 1)
        grad_z = self.compute_gradient(phi, 2)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        grad_mag = np.maximum(grad_mag, self.params.delta_min)
        
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        nz = grad_z / grad_mag
        
        # 表面張力による力の計算
        fx = self.params.surface_tension * kappa * delta * nx
        fy = self.params.surface_tension * kappa * delta * ny
        fz = self.params.surface_tension * kappa * delta * nz
        
        return fx, fy, fz

    def heaviside(self, phi: np.ndarray) -> np.ndarray:
        """ヘビサイド関数の近似"""
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
    
    def advance(self, phi: np.ndarray, velocity: List[np.ndarray], dt: float) -> np.ndarray:
        """時間発展の計算
        
        Args:
            phi: 現在の相場
            velocity: 速度場 [u, v, w]
            dt: 時間刻み幅
            
        Returns:
            更新された相場
        """
        # 移流項
        dphi_dt = np.zeros_like(phi)
        for axis, v in enumerate(velocity):
            bc = self.boundary_conditions.get_condition(axis)
            dphi_dt -= v * self.compute_gradient(phi, axis)
        
        # 界面の拡散項
        mu = self.compute_chemical_potential(phi)
        dphi_dt += self.params.mobility * self.compute_laplacian(mu)
        
        # 表面張力の効果
        fx, fy, fz = self.compute_surface_tension(phi)
        dphi_dt += (fx + fy + fz)
        
        return phi + dt * dphi_dt

    def compute_gradient(self, phi: np.ndarray, axis: int) -> np.ndarray:
        """指定方向の勾配を計算"""
        bc = self.boundary_conditions.get_condition(axis)
        gradient = np.zeros_like(phi)
        
        for idx in self._get_orthogonal_indices(phi.shape, axis):
            line = self._get_line(phi, axis, idx)
            grad_line = self.scheme.apply(line, bc)
            self._set_line(gradient, axis, idx, grad_line)
        
        return gradient

    def _create_grid(self, shape: Tuple[int, ...], 
                    domain_size: Tuple[float, ...]) -> Tuple[np.ndarray, ...]:
        """計算グリッドの生成"""
        x = np.linspace(0, domain_size[0], shape[0])
        y = np.linspace(0, domain_size[1], shape[1])
        z = np.linspace(0, domain_size[2], shape[2])
        return np.meshgrid(x, y, z, indexing='ij')

    def _get_orthogonal_indices(self, shape: Tuple[int, ...], axis: int):
        """指定軸に垂直な方向のインデックスを生成"""
        ranges = [range(s) for i, s in enumerate(shape) if i != axis]
        return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(len(ranges), -1).T
    
    def _get_line(self, array: np.ndarray, axis: int, idx) -> np.ndarray:
        """指定軸に沿ったラインを取得"""
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        return array[tuple(idx_list)]
    
    def _set_line(self, array: np.ndarray, axis: int, idx, values: np.ndarray):
        """指定軸に沿ってラインを設定"""
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        array[tuple(idx_list)] = values