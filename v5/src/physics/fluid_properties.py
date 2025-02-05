from dataclasses import dataclass
from typing import Dict, Callable
import numpy as np

@dataclass
class FluidProperties:
    name: str
    density: float
    viscosity: float

class MultiPhaseProperties:
    def __init__(self, fluids: Dict[str, FluidProperties]):
        self.fluids = fluids
        
    def get_density(self, phase_indicator: np.ndarray) -> np.ndarray:
        """相関数から密度を計算"""
        # 相のリストから密度を取得
        densities = np.array([fluid.density for fluid in self.fluids.values()])
        
        # 相関数の範囲を確認
        phase_indicator = np.clip(phase_indicator, 0, 1)
        
        # 各相の密度の重み付け平均を計算
        density_field = np.zeros_like(phase_indicator, dtype=float)
        density_field += densities[0] * (1 - phase_indicator)  # 第1相
        density_field += densities[1] * phase_indicator        # 第2相
        
        return density_field
    
    def get_viscosity(self, phase_indicator: np.ndarray) -> np.ndarray:
        """相関数から粘度を計算"""
        # 相のリストから粘度を取得
        viscosities = np.array([fluid.viscosity for fluid in self.fluids.values()])
        
        # 相関数の範囲を確認
        phase_indicator = np.clip(phase_indicator, 0, 1)
        
        # 各相の粘度の重み付け平均を計算
        viscosity_field = np.zeros_like(phase_indicator, dtype=float)
        viscosity_field += viscosities[0] * (1 - phase_indicator)  # 第1相
        viscosity_field += viscosities[1] * phase_indicator        # 第2相
        
        return viscosity_field
    
    def _interpolate_property(self, properties: np.ndarray, 
                             phase_indicator: np.ndarray,
                             method: str = 'linear') -> np.ndarray:
        """
        プロパティの補間（廃止予定）
        get_density, get_viscosityメソッドに置き換え
        """
        if method == 'linear':
            # 相関数の範囲を確認
            phase_indicator = np.clip(phase_indicator, 0, 1)
            
            # 各相のプロパティの重み付け平均を計算
            return properties[0] * (1 - phase_indicator) + properties[1] * phase_indicator
        else:
            raise ValueError(f"Unknown interpolation method: {method}")