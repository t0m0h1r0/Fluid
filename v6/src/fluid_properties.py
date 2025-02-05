# physics/fluid_properties.py
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

@dataclass
class FluidComponent:
    """
    単一流体成分の物理的特性を表現するクラス
    
    各流体成分の詳細な物理特性を管理
    """
    name: str
    density: float
    viscosity: float
    
    # オプショナルな追加物理特性
    molecular_weight: Optional[float] = None
    surface_tension: Optional[float] = None
    thermal_conductivity: Optional[float] = None
    specific_heat_capacity: Optional[float] = None
    
    def __post_init__(self):
        """
        初期化後のバリデーション
        """
        if self.density <= 0:
            raise ValueError(f"{self.name}の密度は正の値である必要があります")
        if self.viscosity < 0:
            raise ValueError(f"{self.name}の粘性係数は非負である必要があります")

class MultiComponentFluidProperties:
    """
    多成分流体の物性値を管理・計算するクラス
    
    複数の流体成分間の相互作用と物性値の補間を提供
    """
    def __init__(self, components: List[FluidComponent]):
        """
        多成分流体プロパティの初期化
        
        Args:
            components: 流体成分のリスト
        """
        self.components = {comp.name: comp for comp in components}
        
        # キャッシュと内部状態
        self._volume_fractions_cache: Dict[str, np.ndarray] = {}
    
    def get_density(self, 
                   volume_fractions: Dict[str, np.ndarray], 
                   method: str = 'linear') -> np.ndarray:
        """
        体積分率に基づく密度の計算
        
        Args:
            volume_fractions: 各成分の体積分率
            method: 補間方法 ('linear', 'harmonic', 'arithmetic')
        
        Returns:
            密度場
        """
        # 入力の検証
        self._validate_volume_fractions(volume_fractions)
        
        # 密度の計算
        if method == 'linear':
            return self._linear_density_interpolation(volume_fractions)
        elif method == 'harmonic':
            return self._harmonic_density_interpolation(volume_fractions)
        elif method == 'arithmetic':
            return self._arithmetic_density_interpolation(volume_fractions)
        else:
            raise ValueError(f"未サポートの補間方法: {method}")
    
    def get_viscosity(self, 
                     volume_fractions: Dict[str, np.ndarray], 
                     method: str = 'linear') -> np.ndarray:
        """
        体積分率に基づく粘性係数の計算
        
        Args:
            volume_fractions: 各成分の体積分率
            method: 補間方法 ('linear', 'harmonic', 'arithmetic')
        
        Returns:
            粘性係数場
        """
        # 入力の検証
        self._validate_volume_fractions(volume_fractions)
        
        # 粘性係数の計算
        if method == 'linear':
            return self._linear_viscosity_interpolation(volume_fractions)
        elif method == 'harmonic':
            return self._harmonic_viscosity_interpolation(volume_fractions)
        elif method == 'arithmetic':
            return self._arithmetic_viscosity_interpolation(volume_fractions)
        else:
            raise ValueError(f"未サポートの補間方法: {method}")
    
    def _validate_volume_fractions(self, 
                                   volume_fractions: Dict[str, np.ndarray]):
        """
        体積分率の検証
        
        Args:
            volume_fractions: 各成分の体積分率
        
        Raises:
            ValueError: 検証に失敗した場合
        """
        # 未知の成分のチェック
        unknown_components = set(volume_fractions.keys()) - set(self.components.keys())
        if unknown_components:
            raise ValueError(f"未知の成分: {unknown_components}")
        
        # 体積分率の範囲チェック
        for name, fractions in volume_fractions.items():
            if np.any((fractions < 0) | (fractions > 1)):
                raise ValueError(f"{name}の体積分率は0から1の間である必要があります")
    
    def _linear_density_interpolation(self, 
                                     volume_fractions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        線形的な密度補間
        """
        density = np.zeros_like(list(volume_fractions.values())[0], dtype=float)
        for name, fractions in volume_fractions.items():
            density += fractions * self.components[name].density
        return density
    
    def _harmonic_density_interpolation(self, 
                                       volume_fractions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        調和平均による密度補間
        """
        density_inv = np.zeros_like(list(volume_fractions.values())[0], dtype=float)
        for name, fractions in volume_fractions.items():
            density_inv += fractions / self.components[name].density
        return 1.0 / density_inv
    
    def _arithmetic_density_interpolation(self, 
                                         volume_fractions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        算術平均による密度補間
        """
        return self._linear_density_interpolation(volume_fractions)
    
    def _linear_viscosity_interpolation(self, 
                                       volume_fractions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        線形的な粘性係数補間
        """
        viscosity = np.zeros_like(list(volume_fractions.values())[0], dtype=float)
        for name, fractions in volume_fractions.items():
            viscosity += fractions * self.components[name].viscosity
        return viscosity
    
    def _harmonic_viscosity_interpolation(self, 
                                         volume_fractions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        調和平均による粘性係数補間
        """
        viscosity_inv = np.zeros_like(list(volume_fractions.values())[0], dtype=float)
        for name, fractions in volume_fractions.items():
            viscosity_inv += fractions / self.components[name].viscosity
        return 1.0 / viscosity_inv
    
    def _arithmetic_viscosity_interpolation(self, 
                                           volume_fractions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        算術平均による粘性係数補間
        """
        return self._linear_viscosity_interpolation(volume_fractions)
    
    def compute_surface_tension(self, 
                                volume_fractions: Dict[str, np.ndarray],
                                method: str = 'linear') -> float:
        """
        表面張力の計算
        
        Args:
            volume_fractions: 各成分の体積分率
            method: 補間方法
        
        Returns:
            表面張力係数
        """
        # 入力の検証
        self._validate_volume_fractions(volume_fractions)
        
        surface_tensions = []
        total_fraction = 0.0
        
        for name, fraction in volume_fractions.items():
            component = self.components[name]
            if component.surface_tension is not None:
                surface_tensions.append(component.surface_tension * np.mean(fraction))
                total_fraction += np.mean(fraction)
        
        if not surface_tensions:
            raise ValueError("表面張力係数が定義されていません")
        
        if method == 'linear':
            return sum(surface_tensions) / total_fraction
        elif method == 'harmonic':
            return total_fraction / sum(1.0 / st for st in surface_tensions)
        else:
            raise ValueError(f"未サポートの補間方法: {method}")

# 使用例
def fluid_properties_example():
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
    
    # 3Dグリッドの作成（例）
    Nx, Ny, Nz = 32, 32, 64
    water_fraction = np.random.rand(Nx, Ny, Nz)
    air_fraction = 1.0 - water_fraction
    
    # 密度と粘性係数の計算
    volume_fractions = {
        "water": water_fraction,
        "air": air_fraction
    }
    
    density = fluid_properties.get_density(volume_fractions)
    viscosity = fluid_properties.get_viscosity(volume_fractions)
    surface_tension = fluid_properties.compute_surface_tension(volume_fractions)
    
    print("密度場の形状:", density.shape)
    print("密度場の範囲:", density.min(), "-", density.max())
    print("粘性係数場の形状:", viscosity.shape)
    print("表面張力係数:", surface_tension)

# メイン実行
if __name__ == "__main__":
    fluid_properties_example()
