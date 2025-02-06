import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

@dataclass
class PhaseFieldParameters:
    """Phase-Field法のパラメータ"""
    epsilon: float = 0.01        # 界面厚さ
    mobility: float = 0.1         # 移動度（値を小さく）
    surface_tension: float = 0.07  # 表面張力係数
    double_well_height: float = 1.0  # 二重井戸ポテンシャルの高さ
    stability_factor: float = 1e-6  # 数値安定化のための小さな項

class PhaseField:
    """Phase-Field法による界面追跡"""
    
    def __init__(self, 
                 phase_field: ScalarField,
                 parameters: PhaseFieldParameters):
        self.phi = phase_field  # 秩序変数
        self.params = parameters
        self._initialize_energy_functions()

    def _initialize_energy_functions(self):
        """エネルギー関数の初期化"""
        # 数値安定性を考慮した二重井戸ポテンシャル
        def safe_square(x):
            # オーバーフロー防止のため、値を制限
            x_clipped = np.clip(x, -1e3, 1e3)
            return x_clipped**2
        
        # 変分導関数の安全な計算
        def safe_df(phi):
            phi_clipped = np.clip(phi, -1.0, 1.0)
            return 4.0 * phi_clipped * (phi_clipped**2 - 1.0)
        
        self.f = lambda phi: safe_square(phi**2 - 1.0)
        self.df = safe_df

    def compute_chemical_potential(self) -> np.ndarray:
        """化学ポテンシャルの計算
        μ = df/dφ - ε²∇²φ
        """
        phi = self.phi.data
        dx = self.phi.dx
        eps2 = self.params.epsilon**2
        
        # 変分項
        mu = self.df(phi)
        
        # ラプラシアン項（数値安定性を考慮）
        laplacian = np.zeros_like(phi)
        for i in range(3):
            grad = np.gradient(phi, dx[i], axis=i)
            laplacian += np.gradient(grad, dx[i], axis=i)
        
        # 安定化項を追加
        mu -= eps2 * (laplacian + self.params.stability_factor * phi)
        
        return mu

    def compute_interface_normal(self) -> list[np.ndarray]:
        """界面の法線ベクトルの計算
        n = ∇φ/|∇φ|
        """
        phi = self.phi.data
        dx = self.phi.dx
        
        # 勾配の計算
        grad_phi = [
            np.gradient(phi, dx[i], axis=i)
            for i in range(3)
        ]
        
        # 勾配の大きさ
        grad_norm = np.sqrt(sum(g**2 for g in grad_phi))
        grad_norm = np.maximum(grad_norm, 1e-10)  # ゼロ除算防止
        
        # 正規化
        return [g/grad_norm for g in grad_phi]

    def compute_curvature(self) -> np.ndarray:
        """界面の曲率の計算
        κ = -∇・n
        """
        normal = self.compute_interface_normal()
        dx = self.phi.dx
        
        # 発散の計算
        div_n = np.zeros_like(self.phi.data)
        for i in range(3):
            div_n += np.gradient(normal[i], dx[i], axis=i)
            
        return -div_n

    def compute_surface_tension_force(self) -> list[np.ndarray]:
        """表面張力の計算
        F = σκδ(φ)n
        """
        # デルタ関数
        delta = self.compute_delta_function()
        
        # 曲率
        kappa = self.compute_curvature()
        
        # 法線ベクトル
        normal = self.compute_interface_normal()
        
        # 表面張力
        sigma = self.params.surface_tension
        return [
            sigma * kappa * delta * n 
            for n in normal
        ]

    def compute_delta_function(self) -> np.ndarray:
        """デルタ関数の計算"""
        phi = self.phi.data
        eps = self.params.epsilon
        
        # 数値安定性を考慮したデルタ関数
        return (1.0 / (2.0 * eps)) * np.maximum(
            1.0 - np.tanh(phi/eps)**2, 
            1e-10
        )

    def compute_heaviside_function(self) -> np.ndarray:
        """ヘビサイド関数の計算"""
        phi = self.phi.data
        eps = self.params.epsilon
        
        # クリッピングを追加
        phi_clipped = np.clip(phi, -1.0, 1.0)
        return 0.5 * (1.0 + np.tanh(phi_clipped/eps))

    def compute_mixing_energy(self) -> float:
        """混合エネルギーの計算"""
        phi = self.phi.data
        eps = self.params.epsilon
        
        # 勾配エネルギー
        grad_energy = 0.0
        for i in range(3):
            grad_phi = np.gradient(phi, self.phi.dx[i], axis=i)
            grad_energy += np.sum(grad_phi**2)
        grad_energy *= 0.5 * eps**2
        
        # 二重井戸ポテンシャル
        well_energy = np.sum(self.f(phi))
        
        return grad_energy + well_energy

    def get_interface_location(self, threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """界面位置の取得"""
        from skimage import measure
        
        phi = self.phi.data
        dx = self.phi.dx
        
        try:
            verts, faces, _, _ = measure.marching_cubes(
                phi,
                level=threshold,
                spacing=dx
            )
            return verts[:,0], verts[:,1], verts[:,2]
            
        except:
            # 界面が見つからない場合は空の配列を返す
            return np.array([]), np.array([]), np.array([])

    def evolve(self, velocity_field: Optional[VectorField] = None, dt: float = 0.001) -> None:
        """Phase-Fieldの時間発展"""
        phi = self.phi.data
        dx = self.phi.dx
        
        # 化学ポテンシャル
        mu = self.compute_chemical_potential()
        
        # Cahn-Hilliard方程式の右辺
        dphi_dt = self.params.mobility * sum(
            np.gradient(
                np.gradient(mu, dx[i], axis=i),
                dx[i], axis=i
            )
            for i in range(3)
        )
        
        # 移流項の追加（数値安定性を考慮）
        if velocity_field is not None:
            for i, v in enumerate(velocity_field.data):
                # 移流項の計算時に値をクリッピング
                grad = np.gradient(phi, dx[i], axis=i)
                dphi_dt -= np.clip(v, -1e3, 1e3) * grad
        
        # 時間発展（値をクリッピング）
        self.phi.data = np.clip(phi + dt * dphi_dt, -1.0, 1.0)