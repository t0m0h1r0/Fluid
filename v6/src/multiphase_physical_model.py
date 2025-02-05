# physics/multiphase_model.py
import numpy as np
from core.interfaces import PhysicalModel
from typing import Dict, Any
from physics.fluid_properties import MultiComponentFluidProperties

class MultiphaseNavierStokesModel(PhysicalModel):
    """
    多相流体のNavier-Stokes方程式モデル
    
    移流、拡散、外力項を含む複合的な物理モデル
    """
    def __init__(self, 
                 fluid_properties: MultiComponentFluidProperties,
                 gravity: float = 9.81):
        """
        多相流体モデルの初期化
        
        Args:
            fluid_properties: 流体物性値マネージャ
            gravity: 重力加速度
        """
        self.fluid_properties = fluid_properties
        self.gravity = gravity
    
    def compute_flux(self, 
                     velocity: np.ndarray, 
                     parameters: Dict[str, Any]) -> np.ndarray:
        """
        移流項の計算
        
        Args:
            velocity: 速度場
            parameters: 追加パラメータ（密度、位相場など）
        
        Returns:
            移流項のフラックス
        """
        # 密度と位相場の取得
        phase_field = parameters.get('phase_field')
        
        if phase_field is None:
            raise ValueError("位相場パラメータが必要です")
        
        # 体積分率の計算
        volume_fractions = self._compute_volume_fractions(phase_field)
        
        # 密度の計算
        density = self.fluid_properties.get_density(volume_fractions)
        
        # 移流項の計算（簡易的なWENOスキーム）
        flux = np.zeros_like(velocity)
        
        for axis in range(velocity.ndim):
            # 風上差分の方向を決定
            upwind = velocity[axis] < 0
            
            # 風上差分の近似
            flux[axis] = self._upwind_flux_approximation(
                velocity[axis], 
                density, 
                upwind
            )
        
        return flux
    
    def compute_source_terms(self, 
                             velocity: np.ndarray, 
                             parameters: Dict[str, Any]) -> np.ndarray:
        """
        外力項の計算
        
        Args:
            velocity: 速度場
            parameters: 追加パラメータ（密度、位相場など）
        
        Returns:
            外力項
        """
        # 密度と位相場の取得
        phase_field = parameters.get('phase_field')
        
        if phase_field is None:
            raise ValueError("位相場パラメータが必要です")
        
        # 体積分率の計算
        volume_fractions = self._compute_volume_fractions(phase_field)
        
        # 密度の計算
        density = self.fluid_properties.get_density(volume_fractions)
        
        # 外力項の計算（重力）
        source = [np.zeros_like(velocity[0]) for _ in range(velocity.ndim)]
        
        # z軸方向（鉛直方向）の重力項
        source[2] = -self.gravity * density
        
        return source
    
    def _compute_volume_fractions(self, phase_field: np.ndarray) -> Dict[str, np.ndarray]:
        """
        位相場から体積分率を計算
        
        Args:
            phase_field: 位相場
        
        Returns:
            各成分の体積分率
        """
        # ヘビサイド関数による重み付け
        def heaviside(phi: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
            return 0.5 * (1.0 + np.tanh(phi / epsilon))
        
        # 初期化
        volume_fractions = {}
        
        # 現在は2相流体を想定（拡張可能）
        components = list(self.fluid_properties.components.keys())
        
        # 第1成分（例：水）
        volume_fractions[components[0]] = heaviside(phase_field)
        
        # 第2成分（例：空気）
        volume_fractions[components[1]] = 1.0 - volume_fractions[components[0]]
        
        return volume_fractions
    
    def _upwind_flux_approximation(self, 
                                   velocity: np.ndarray, 
                                   density: np.ndarray, 
                                   upwind: np.ndarray) -> np.ndarray:
        """
        風上差分による移流項の近似
        
        Args:
            velocity: 速度成分
            density: 密度場
            upwind: 風上方向のブール配列
        
        Returns:
            移流項のフラックス
        """
        # 風上差分の近似
        flux = np.zeros_like(velocity)
        
        # 風上側の値を選択
        flux_forward = np.roll(velocity, -1, axis=0)
        flux_backward = np.roll(velocity, 1, axis=0)
        
        # 風上差分の計算
        flux[upwind] = flux_backward[upwind]
        flux[~upwind] = flux_forward[~upwind]
        
        # 密度との積
        flux *= density
        
        return flux

class PhaseFieldModel(PhysicalModel):
    """
    位相場の時間発展モデル
    
    界面追跡と相変化のためのPhase-Field方法
    """
    def __init__(self, 
                 interface_width: float = 0.01, 
                 mobility: float = 1.0):
        """
        位相場モデルの初期化
        
        Args:
            interface_width: 界面の厚さ
            mobility: 界面移動の係数
        """
        self.interface_width = interface_width
        self.mobility = mobility
    
    def compute_flux(self, 
                     phase_field: np.ndarray, 
                     parameters: Dict[str, Any]) -> np.ndarray:
        """
        位相場の移流項の計算
        
        Args:
            phase_field: 位相場
            parameters: 追加パラメータ（速度場など）
        
        Returns:
            位相場の移流項のフラックス
        """
        # 速度場の取得
        velocity = parameters.get('velocity')
        
        if velocity is None:
            raise ValueError("速度場パラメータが必要です")
        
        # 位相場の移流項計算
        flux = np.zeros_like(phase_field)
        
        for axis in range(phase_field.ndim):
            # 風上差分の方向を決定
            upwind = velocity[axis] < 0
            
            # 風上差分の近似
            flux_forward = np.roll(phase_field, -1, axis)
            flux_backward = np.roll(phase_field, 1, axis)
            
            # 風上差分の計算
            flux_component = np.where(
                upwind, 
                flux_backward, 
                flux_forward
            )
            
            # 移流項の計算
            flux += velocity[axis] * (flux_component - phase_field)
        
        return flux
    
    def compute_source_terms(self, 
                             phase_field: np.ndarray, 
                             parameters: Dict[str, Any]) -> np.ndarray:
        """
        位相場のソース項の計算（化学ポテンシャル項）
        
        Args:
            phase_field: 位相場
            parameters: 追加パラメータ
        
        Returns:
            位相場のソース項
        """
        # 化学ポテンシャルの計算
        chemical_potential = self._compute_chemical_potential(phase_field)
        
        # モビリティとの積
        return self.mobility * chemical_potential
    
    def _compute_chemical_potential(self, phase_field: np.ndarray) -> np.ndarray:
        """
        位相場の化学ポテンシャルの計算
        
        Args:
            phase_field: 位相場
        
        Returns:
            化学ポテンシャル
        """
        # 勾配エネルギー項
        def compute_laplacian(field: np.ndarray) -> np.ndarray:
            """簡易的なラプラシアンの計算"""
            laplacian = np.zeros_like(field)
            for axis in range(field.ndim):
                laplacian += (
                    np.roll(field, -1, axis) - 
                    2 * field + 
                    np.roll(field, 1, axis)
                )
            return laplacian
        
        # 勾配エネルギー項
        gradient_energy = -self.interface_width**2 * compute_laplacian(phase_field)
        
        # 二重井戸ポテンシャル項
        bulk_energy = phase_field * (phase_field**2 - 1.0)
        
        return gradient_energy + bulk_energy

# デモンストレーション関数
def demonstrate_multiphase_model():
    """
    多相流体モデルのデモンストレーション
    """
    # 流体物性の設定
    from physics.fluid_properties import FluidComponent, MultiComponentFluidProperties
    
    # 流体成分の定義
    water = FluidComponent(
        name="water", 
        density=1000.0, 
        viscosity=1e-3, 
        surface_tension=0.072
    )
    air = FluidComponent(
        name="air", 
        density=1.225, 
        viscosity=1.81e-5, 
        surface_tension=0.03
    )
    
    # 多成分流体プロパティの初期化
    fluid_properties = MultiComponentFluidProperties([water, air])
    
    # 3Dグリッドの作成
    Nx, Ny, Nz = 32, 32, 64
    
    # 初期位相場（界面を含む）
    phase_field = np.random.uniform(-1, 1, (Nx, Ny, Nz))
    
    # 速度場の初期化
    velocity = [np.random.rand(Nx, Ny, Nz) for _ in range(3)]
    
    # モデルの初期化
    navier_stokes_model = MultiphaseNavierStokesModel(fluid_properties)
    phase_field_model = PhaseFieldModel()
    
    # フラックスの計算
    ns_flux = navier_stokes_model.compute_flux(velocity[0], {'phase_field': phase_field})
    pf_flux = phase_field_model.compute_flux(phase_field, {'velocity': velocity})
    
    # ソース項の計算
    ns_source = navier_stokes_model.compute_source_terms(velocity[0], {'phase_field': phase_field})
    pf_source = phase_field_model.compute_source_terms(phase_field, {})
    
    # 結果の表示
    print("Navier-Stokesモデル:")
    print(f"  フラックス形状: {ns_flux.shape}")
    print(f"  フラックス範囲: [{ns_flux.min():.4f}, {ns_flux.max():.4f}]")
    print(f"  ソース項形状: {ns_source[2].shape}")
    print(f"  ソース項範囲: [{ns_source[2].min():.4f}, {ns_source[2].max():.4f}]")
    
    print("\nPhase-Fieldモデル:")
    print(f"  フラックス形状: {pf_flux.shape}")
    print(f"  フラックス範囲: [{pf_flux.min():.4f}, {pf_flux.max():.4f}]")
    print(f"  ソース項形状: {pf_source.shape}")
    print(f"  ソース項範囲: [{pf_source.min():.4f}, {pf_source.max():.4f}]")

# メイン実行
if __name__ == "__main__":
    demonstrate_multiphase_model()
