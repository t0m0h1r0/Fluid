"""流体の物性値を管理するモジュールを提供します。

このモジュールは、二相流体シミュレーションにおける物性値（密度、粘性係数、表面張力係数）を
管理し、Level Set関数との連携による界面での適切な物性値の計算を行います。
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal
import numpy as np
from physics.levelset import LevelSetField, heaviside

@dataclass
class FluidProperties:
    """流体の物性値を保持するクラス
    
    Attributes:
        density: 密度 [kg/m³]
        viscosity: 動粘性係数 [Pa·s]
        surface_tension: 表面張力係数 [N/m]
    """
    density: float
    viscosity: float
    surface_tension: Optional[float] = None
    
    def __post_init__(self):
        """物性値の妥当性チェック"""
        if self.density <= 0:
            raise ValueError("密度は正の値である必要があります")
        if self.viscosity <= 0:
            raise ValueError("粘性係数は正の値である必要があります")
        if self.surface_tension is not None and self.surface_tension < 0:
            raise ValueError("表面張力係数は非負である必要があります")

class PropertiesManager:
    """流体の物性値を管理するクラス
    
    二相流体の物性値を管理し、Level Set関数に基づいて
    界面での適切な物性値の計算を行います。
    """
    
    def __init__(self, 
                 phase1: FluidProperties,
                 phase2: FluidProperties,
                 interpolation_method: Literal['arithmetic', 'harmonic'] = 'arithmetic'):
        """物性値マネージャーを初期化
        
        Args:
            phase1: 第1相の物性値（Level Set関数が正の領域）
            phase2: 第2相の物性値（Level Set関数が負の領域）
            interpolation_method: 物性値の補間方法
        """
        self.phase1 = phase1
        self.phase2 = phase2
        self.interpolation_method = interpolation_method
        
        # 表面張力係数の設定
        self.surface_tension = None
        if phase1.surface_tension is not None and phase2.surface_tension is not None:
            # 両方の相で定義されている場合は平均を取る
            self.surface_tension = 0.5 * (phase1.surface_tension + phase2.surface_tension)
        elif phase1.surface_tension is not None:
            self.surface_tension = phase1.surface_tension
        elif phase2.surface_tension is not None:
            self.surface_tension = phase2.surface_tension
        
        # キャッシュの初期化
        self._cache: Dict[str, np.ndarray] = {}
    
    def get_density(self, phi: LevelSetField) -> np.ndarray:
        """密度場を計算
        
        Args:
            phi: Level Set場
            
        Returns:
            密度場
        """
        cache_key = ('density', id(phi))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Heaviside関数による補間
        H = heaviside(phi.data, phi.params.epsilon)
        density = self.phase1.density * H + self.phase2.density * (1 - H)
        
        self._cache[cache_key] = density
        return density
    
    def get_viscosity(self, phi: LevelSetField) -> np.ndarray:
        """粘性係数場を計算
        
        Args:
            phi: Level Set場
            
        Returns:
            粘性係数場
        """
        cache_key = ('viscosity', id(phi))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self.interpolation_method == 'arithmetic':
            # 算術平均
            H = heaviside(phi.data, phi.params.epsilon)
            viscosity = self.phase1.viscosity * H + self.phase2.viscosity * (1 - H)
        else:
            # 調和平均（粘性に対してより適切）
            H = heaviside(phi.data, phi.params.epsilon)
            viscosity = 1.0 / (H / self.phase1.viscosity + 
                             (1 - H) / self.phase2.viscosity)
        
        self._cache[cache_key] = viscosity
        return viscosity
    
    def get_kinematic_viscosity(self, phi: LevelSetField) -> np.ndarray:
        """動粘性係数場を計算
        
        Args:
            phi: Level Set場
            
        Returns:
            動粘性係数場
        """
        return self.get_viscosity(phi) / self.get_density(phi)
    
    def get_surface_tension_coefficient(self) -> Optional[float]:
        """表面張力係数を取得
        
        Returns:
            表面張力係数（未定義の場合はNone）
        """
        return self.surface_tension
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self._cache.clear()
    
    def get_property_jump(self, property_name: str) -> float:
        """物性値のジャンプ（不連続性）を取得
        
        Args:
            property_name: 物性値の名前（'density' または 'viscosity'）
            
        Returns:
            物性値のジャンプの大きさ
        """
        if property_name == 'density':
            return abs(self.phase1.density - self.phase2.density)
        elif property_name == 'viscosity':
            return abs(self.phase1.viscosity - self.phase2.viscosity)
        else:
            raise ValueError(f"未知の物性値: {property_name}")
    
    def get_all_properties(self, phi: LevelSetField) -> Dict[str, np.ndarray]:
        """全ての物性値場を取得
        
        Args:
            phi: Level Set場
            
        Returns:
            物性値場の辞書
        """
        return {
            'density': self.get_density(phi),
            'viscosity': self.get_viscosity(phi),
            'kinematic_viscosity': self.get_kinematic_viscosity(phi)
        }
    
    def get_diagnostics(self, phi: LevelSetField) -> Dict[str, Any]:
        """物性値の診断情報を取得
        
        Args:
            phi: Level Set場
            
        Returns:
            診断情報の辞書
        """
        density = self.get_density(phi)
        viscosity = self.get_viscosity(phi)
        
        return {
            'density': {
                'min': np.min(density),
                'max': np.max(density),
                'mean': np.mean(density),
                'jump': self.get_property_jump('density')
            },
            'viscosity': {
                'min': np.min(viscosity),
                'max': np.max(viscosity),
                'mean': np.mean(viscosity),
                'jump': self.get_property_jump('viscosity')
            },
            'surface_tension': self.surface_tension,
            'interpolation_method': self.interpolation_method
        }