# physics/phase_field.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from core.scheme import DifferenceScheme
from core.boundary import DirectionalBC
from scipy.ndimage import gaussian_filter

@dataclass
class PhaseFieldParams:
    """相場パラメータ"""
    epsilon: float = 0.01        # 界面厚さ
    mobility: float = 1.0        # 移動度
    surface_tension: float = 0.07  # 表面張力係数
    delta_min: float = 1e-6      # デルタ関数の最小値
    kappa_max: float = 100.0     # 曲率の最大値
    stabilization_factor: float = 0.1  # 数値安定化係数

@dataclass
class PhaseObject:
    """相場オブジェクトの基底クラス"""
    phase: str  # phase name

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
        self._initialize_stabilization()

    def _initialize_stabilization(self):
        """数値安定化パラメータの初期化"""
        self._stab_coeff = self.params.stabilization_factor
        self._phase_bounds = (-1.0, 1.0)  # 相変数の物理的な範囲

    def initialize_field(self, shape: Tuple[int, ...], 
                        domain_size: Tuple[float, ...]) -> np.ndarray:
        """相場の初期化"""
        self._grid = self._create_grid(shape, domain_size)
        # 全体を低密度相（-1）で初期化
        return -np.ones(shape)

    def set_phase_density(self, phase: str, density: float):
        """相の密度を登録"""
        self._phase_densities[phase] = density

    def add_layer(self, field: np.ndarray, layer: Layer) -> np.ndarray:
        """レイヤーを相場に追加"""
        if self._grid is None:
            raise RuntimeError("グリッドが初期化されていません。initialize_fieldを先に呼び出してください。")
        
        X, Y, Z = self._grid
        z_min, z_max = layer.z_range
        mask = (Z >= z_min) & (Z < z_max)
        
        if layer.phase not in self._phase_densities:
            raise ValueError(f"相 {layer.phase} の密度が設定されていません")
        
        # 相変数を[-1, 1]の範囲で設定
        # 密度の高い相を1、低い相を-1とする
        max_density = max(self._phase_densities.values())
        min_density = min(self._phase_densities.values())
        
        if self._phase_densities[layer.phase] == max_density:
            field[mask] = 1.0
        else:
            field[mask] = -1.0
        
        return field

    def add_sphere(self, field: np.ndarray, sphere: Sphere) -> np.ndarray:
        """球を相場に追加"""
        if self._grid is None:
            raise RuntimeError("グリッドが初期化されていません。initialize_fieldを先に呼び出してください。")
            
        X, Y, Z = self._grid
        r = np.sqrt(
            (X - sphere.center[0])**2 + 
            (Y - sphere.center[1])**2 + 
            (Z - sphere.center[2])**2
        )
        mask = r <= sphere.radius
        
        if sphere.phase not in self._phase_densities:
            raise ValueError(f"相 {sphere.phase} の密度が設定されていません")
        
        # 相変数を[-1, 1]の範囲で設定
        max_density = max(self._phase_densities.values())
        min_density = min(self._phase_densities.values())
        
        if self._phase_densities[sphere.phase] == max_density:
            field[mask] = 1.0
        else:
            field[mask] = -1.0
        
        return field

    def advance(self, phi: np.ndarray, velocity: List[np.ndarray], dt: float) -> np.ndarray:
        """改良された時間発展の計算"""
        # 移流項
        dphi_dt = self._compute_advection(phi, velocity)
        
        # Cahn-Hilliard項
        mu = self.compute_chemical_potential(phi)
        dphi_dt += self.params.mobility * self.compute_laplacian(mu)
        
        # 表面張力項
        fx, fy, fz = self.compute_surface_tension(phi)
        dphi_dt += (fx + fy + fz)
        
        # 数値安定化項
        dphi_dt += self._compute_stabilization(phi)
        
        # 時間積分
        phi_new = phi + dt * dphi_dt
        
        # 物理的な制約の適用
        return self._apply_physical_constraints(phi_new)

    def _compute_advection(self, phi: np.ndarray, velocity: List[np.ndarray]) -> np.ndarray:
        """WENOスキームを用いた移流項の計算"""
        dphi_dt = np.zeros_like(phi)
        
        for axis, v in enumerate(velocity):
            # 風上差分の方向を決定
            upwind = v < 0
            
            # WENO5による風上差分
            flux_plus = self._weno5(phi, axis, False)
            flux_minus = self._weno5(phi, axis, True)
            
            # フラックスの合成
            flux = np.where(upwind, flux_minus, flux_plus)
            dphi_dt -= v * flux
        
        return dphi_dt

    def _weno5(self, field: np.ndarray, axis: int, is_negative: bool) -> np.ndarray:
        """5次精度WENOスキームの実装"""
        # WENOの重み係数
        epsilon = 1e-6
        gamma0, gamma1, gamma2 = 0.1, 0.6, 0.3
        
        # シフトしたインデックスを準備
        if is_negative:
            v1 = np.roll(field, 2, axis=axis)
            v2 = np.roll(field, 1, axis=axis)
            v3 = field
            v4 = np.roll(field, -1, axis=axis)
            v5 = np.roll(field, -2, axis=axis)
        else:
            v1 = np.roll(field, -2, axis=axis)
            v2 = np.roll(field, -1, axis=axis)
            v3 = field
            v4 = np.roll(field, 1, axis=axis)
            v5 = np.roll(field, 2, axis=axis)
        
        # 3つの候補ステンシル
        s0 = (13/12) * (v1 - 2*v2 + v3)**2 + (1/4) * (v1 - 4*v2 + 3*v3)**2
        s1 = (13/12) * (v2 - 2*v3 + v4)**2 + (1/4) * (v2 - v4)**2
        s2 = (13/12) * (v3 - 2*v4 + v5)**2 + (1/4) * (3*v3 - 4*v4 + v5)**2
        
        # 非線形重み
        alpha0 = gamma0 / (epsilon + s0)**2
        alpha1 = gamma1 / (epsilon + s1)**2
        alpha2 = gamma2 / (epsilon + s2)**2
        omega = np.array([alpha0, alpha1, alpha2])
        omega /= np.sum(omega, axis=0)
        
        # 各ステンシルでの補間値
        p0 = (1/6) * (2*v1 - 7*v2 + 11*v3)
        p1 = (1/6) * (-v2 + 5*v3 + 2*v4)
        p2 = (1/6) * (2*v3 + 5*v4 - v5)
        
        return omega[0]*p0 + omega[1]*p1 + omega[2]*p2

    def compute_chemical_potential(self, phi: np.ndarray) -> np.ndarray:
        """改良された化学ポテンシャルの計算"""
        # 勾配エネルギー項
        laplacian = self.compute_laplacian(phi)
        gradient_energy = -self.params.epsilon**2 * laplacian
        
        # 二重井戸ポテンシャル項
        bulk_energy = phi * (phi**2 - 1.0)
        
        # 安定化項
        stabilization = self._stab_coeff * (np.maximum(phi - 1.0, 0.0) - 
                                          np.minimum(phi + 1.0, 0.0))
        
        return bulk_energy + gradient_energy + stabilization

    def compute_surface_tension(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """改良された表面張力の計算"""
        # 界面の法線と曲率の計算
        kappa = self.compute_curvature(phi)
        grad_x = self.compute_gradient(phi, 0)
        grad_y = self.compute_gradient(phi, 1)
        grad_z = self.compute_gradient(phi, 2)
        
        # 勾配の大きさと正規化
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        grad_mag = np.maximum(grad_mag, self.params.delta_min)
        
        nx = grad_x / grad_mag
        ny = grad_y / grad_mag
        nz = grad_z / grad_mag
        
        # デルタ関数の計算と改良
        delta = self.delta(phi)
        delta = self._smooth_delta(delta)
        
        # 表面張力の計算
        coeff = self.params.surface_tension * kappa * delta
        return (coeff * nx, coeff * ny, coeff * nz)

    def compute_curvature(self, phi: np.ndarray) -> np.ndarray:
        """界面の曲率を計算"""
        grad_x = self.compute_gradient(phi, 0)
        grad_y = self.compute_gradient(phi, 1)
        grad_z = self.compute_gradient(phi, 2)
        
        grad_xx = self.compute_gradient(grad_x, 0)
        grad_yy = self.compute_gradient(grad_y, 1)
        grad_zz = self.compute_gradient(grad_z, 2)
        
        grad_xy = self.compute_gradient(grad_x, 1)
        grad_yz = self.compute_gradient(grad_y, 2)
        grad_zx = self.compute_gradient(grad_z, 0)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        grad_mag = np.maximum(grad_mag, self.params.delta_min)
        
        kappa = (
            (grad_xx * (grad_y**2 + grad_z**2) + 
             grad_yy * (grad_z**2 + grad_x**2) + 
             grad_zz * (grad_x**2 + grad_y**2) - 
             2 * (grad_xy * grad_x * grad_y + 
                  grad_yz * grad_y * grad_z + 
                  grad_zx * grad_z * grad_x)
            ) / grad_mag**3
        )
        
        return np.clip(kappa, -self.params.kappa_max, self.params.kappa_max)

    def heaviside(self, phi: np.ndarray) -> np.ndarray:
        """ヘビサイド関数の近似"""
        return 0.5 * (1.0 + np.tanh(phi / self.params.epsilon))
    
    def delta(self, phi: np.ndarray) -> np.ndarray:
        """デルタ関数の近似"""
        return (1.0 / (2.0 * self.params.epsilon)) * (
            1.0 - np.tanh(phi / self.params.epsilon)**2
        )
    
    def _smooth_delta(self, delta: np.ndarray) -> np.ndarray:
        """デルタ関数の数値的な安定化"""
        sigma = 0.5  # 平滑化の程度
        return gaussian_filter(delta, sigma)

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

    def compute_gradient(self, phi: np.ndarray, axis: int) -> np.ndarray:
        """指定方向の勾配を計算"""
        bc = self.boundary_conditions.get_condition(axis)
        gradient = np.zeros_like(phi)
        
        for idx in self._get_orthogonal_indices(phi.shape, axis):
            line = self._get_line(phi, axis, idx)
            grad_line = self.scheme.apply(line, bc)
            self._set_line(gradient, axis, idx, grad_line)
        
        return gradient

    def _apply_physical_constraints(self, phi: np.ndarray) -> np.ndarray:
        """相変数の物理的な制約の適用"""
        # 範囲の制限
        phi = np.clip(phi, self._phase_bounds[0], self._phase_bounds[1])
        
        # 質量保存の補正
        initial_mass = np.sum(self.heaviside(phi))
        current_mass = np.sum(self.heaviside(phi))
        if np.abs(current_mass - initial_mass) > 1e-10:
            phi *= np.sqrt(initial_mass / current_mass)
        
        return phi

    def _compute_stabilization(self, phi: np.ndarray) -> np.ndarray:
        """数値安定化項の計算"""
        # 勾配制限付き拡散項
        grad_mag = np.sqrt(sum(self.compute_gradient(phi, i)**2 for i in range(3)))
        diffusion = self.compute_laplacian(phi)
        
        return -self._stab_coeff * np.tanh(grad_mag) * diffusion

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

    def compute_interface_energy(self, phi: np.ndarray) -> float:
        """界面エネルギーの計算"""
        # 勾配エネルギー
        grad_energy = sum(
            0.5 * self.params.epsilon**2 * np.sum(self.compute_gradient(phi, i)**2)
            for i in range(3)
        )
        
        # 二重井戸ポテンシャルエネルギー
        well_energy = 0.25 * np.sum((phi**2 - 1.0)**2)
        
        return grad_energy + well_energy

    def get_phase_fraction(self, phi: np.ndarray) -> float:
        """相分率の計算"""
        H = self.heaviside(phi)
        return np.sum(H) / phi.size

    def get_interface_thickness(self, phi: np.ndarray) -> float:
        """界面厚さの実効値を計算"""
        delta = self.delta(phi)
        if np.max(delta) > self.params.delta_min:
            return 1.0 / np.max(delta)
        else:
            return float('inf')

    def validate_configuration(self) -> List[str]:
        """設定の妥当性チェック"""
        errors = []
        
        # パラメータの範囲チェック
        if self.params.epsilon <= 0:
            errors.append("界面厚さ（epsilon）は正の値である必要があります")
        if self.params.mobility <= 0:
            errors.append("移動度（mobility）は正の値である必要があります")
        if self.params.surface_tension < 0:
            errors.append("表面張力係数は非負である必要があります")
        
        # 位相密度の設定チェック
        if not self._phase_densities:
            errors.append("位相密度が設定されていません")
        
        # グリッドの初期化チェック
        if self._grid is None:
            errors.append("グリッドが初期化されていません")
        
        return errors

    def get_interface_location(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """界面位置の取得"""
        # 界面はφ=0の等値面として定義
        from skimage import measure
        
        # 等値面の抽出
        try:
            verts, faces, _, _ = measure.marching_cubes(
                phi,
                level=0.0,
                allow_degenerate=False,
                spacing=(1.0, 1.0, 1.0)  # グリッド間隔を指定
            )
            
            if len(verts) > 0:
                return verts[:, 0], verts[:, 1], verts[:, 2]
            
        except ValueError:
            pass
        
        return np.array([]), np.array([]), np.array([])