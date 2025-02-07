from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Optional, Tuple
from core.field import Field

@dataclass
class FluidPhase:
    """流体相の物性値"""
    name: str
    density: float
    viscosity: float
    surface_tension: Optional[float] = None

class FluidProperties:
    """流体物性値の管理クラス"""
    
    def __init__(self, phases: Dict[str, FluidPhase]):
        """
        Args:
            phases: 相の辞書（キーは相の名前）
        """
        self.phases = phases
        self._validate_phases()
        
        # キャッシュの初期化
        self._cache = {}
        
        # デバッグ情報
        print("Initialized FluidProperties with phases:")
        for name, phase in phases.items():
            print(f"  {name}:")
            print(f"    density: {phase.density} kg/m³")
            print(f"    viscosity: {phase.viscosity} Pa·s")
            if phase.surface_tension is not None:
                print(f"    surface tension: {phase.surface_tension} N/m")
    
    def _validate_phases(self):
        """相の妥当性をチェック"""
        for name, phase in self.phases.items():
            if phase.density <= 0:
                raise ValueError(f"相 {name} の密度は正である必要があります: {phase.density}")
            if phase.viscosity <= 0:
                raise ValueError(f"相 {name} の粘性係数は正である必要があります: {phase.viscosity}")
            if phase.surface_tension is not None and phase.surface_tension < 0:
                raise ValueError(f"相 {name} の表面張力係数は非負である必要があります: {phase.surface_tension}")
    
    def get_density(self, phi: Field) -> np.ndarray:
        """密度場の取得"""
        try:
            return self._get_property(phi, lambda phase: phase.density)
        except Exception as e:
            print(f"Error in get_density: {str(e)}")
            print(f"Field shape: {phi.shape}")
            raise
    
    def get_viscosity(self, phi: Field) -> np.ndarray:
        """粘性係数場の取得"""
        try:
            return self._get_property(phi, lambda phase: phase.viscosity)
        except Exception as e:
            print(f"Error in get_viscosity: {str(e)}")
            print(f"Field shape: {phi.shape}")
            raise
    
    def get_surface_tension(self, phi1: Field, phi2: Field) -> float:
        """2相間の表面張力係数を取得"""
        try:
            phase1 = self._get_phase_from_field(phi1)
            phase2 = self._get_phase_from_field(phi2)
            
            # 両方の相で表面張力が定義されている場合は平均を取る
            if phase1.surface_tension is not None and phase2.surface_tension is not None:
                return 0.5 * (phase1.surface_tension + phase2.surface_tension)
            # どちらかの相で定義されている場合はその値を使用
            elif phase1.surface_tension is not None:
                return phase1.surface_tension
            elif phase2.surface_tension is not None:
                return phase2.surface_tension
            # 両方未定義の場合はデフォルト値
            else:
                return 0.07  # 水の表面張力係数をデフォルトとして使用
        except Exception as e:
            print(f"Error in get_surface_tension: {str(e)}")
            print(f"Field shapes - phi1: {phi1.shape}, phi2: {phi2.shape}")
            raise
    
    def _get_property(self, phi: Field, property_func: callable) -> np.ndarray:
        """指定された物性値の場を取得"""
        try:
            # キャッシュキーの生成
            cache_key = (id(phi), property_func.__name__)
            
            # キャッシュにある場合はそれを返す
            if cache_key in self._cache:
                return self._cache[cache_key]
            
            # ヘビサイド関数による重み付け
            H = self._heaviside(phi.data)
            
            # 各相の物性値の配列を作成
            values = np.array([property_func(phase) for phase in self.phases.values()])
            
            # 重み付き平均の計算
            result = np.sum(values[:, np.newaxis, np.newaxis, np.newaxis] * H, axis=0)
            
            # キャッシュに保存
            self._cache[cache_key] = result
            
            return result
        except Exception as e:
            print(f"Error in _get_property: {str(e)}")
            print(f"Field shape: {phi.shape}")
            print(f"Property function: {property_func.__name__}")
            raise
    
    def _get_phase_from_field(self, phi: Field) -> FluidPhase:
        """場から対応する相を取得"""
        try:
            # 最も体積分率の大きい相を選択
            H = self._heaviside(phi.data)
            phase_index = np.argmax(np.mean(H, axis=(1, 2, 3)))
            return list(self.phases.values())[phase_index]
        except Exception as e:
            print(f"Error in _get_phase_from_field: {str(e)}")
            print(f"Field shape: {phi.shape}")
            raise
    
    def _heaviside(self, phi: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """ヘビサイド関数の計算"""
        try:
            result = 0.5 * (1.0 + np.tanh(phi / epsilon))
            print(f"Heaviside function computed, output shape: {result.shape}")
            return result
        except Exception as e:
            print(f"Error in _heaviside: {str(e)}")
            print(f"Input array shape: {phi.shape}")
            raise
    
    def clear_cache(self):
        """キャッシュのクリア"""
        self._cache.clear()
    
    def update_properties(self, phase_name: str, **kwargs):
        """物性値の更新"""
        if phase_name not in self.phases:
            raise ValueError(f"相 {phase_name} は存在しません")
            
        phase = self.phases[phase_name]
        for key, value in kwargs.items():
            if hasattr(phase, key):
                setattr(phase, key, value)
            else:
                raise ValueError(f"無効なプロパティ: {key}")
        
        # プロパティが更新されたのでキャッシュをクリア
        self.clear_cache()
        
    def interpolate_property(self, phi: Field, property_name: str,
                           method: str = 'arithmetic') -> np.ndarray:
        """物性値の補間

        Args:
            phi: Level Set場
            property_name: 物性値の名前
            method: 補間方法（'arithmetic'または'harmonic'）
        """
        try:
            if method not in ['arithmetic', 'harmonic']:
                raise ValueError("補間方法は'arithmetic'または'harmonic'である必要があります")
                
            if property_name not in ['density', 'viscosity']:
                raise ValueError("サポートされていない物性値です")
                
            # プロパティ値の取得
            values = np.array([getattr(phase, property_name) 
                            for phase in self.phases.values()])
            
            # ヘビサイド関数による重み付け
            H = self._heaviside(phi.data)
            
            if method == 'arithmetic':
                # 算術平均
                return np.sum(values[:, np.newaxis, np.newaxis, np.newaxis] * H, axis=0)
            else:
                # 調和平均
                return 1.0 / np.sum(H / values[:, np.newaxis, np.newaxis, np.newaxis], 
                                  axis=0)
        except Exception as e:
            print(f"Error in interpolate_property: {str(e)}")
            print(f"Field shape: {phi.shape}")
            print(f"Property: {property_name}, Method: {method}")
            raise
    
    def get_mixture_properties(self, phi: Field) -> Dict[str, np.ndarray]:
        """全ての物性値を取得"""
        try:
            properties = {
                'density': self.get_density(phi),
                'viscosity': self.get_viscosity(phi)
            }
            print(f"Mixture properties computed - shapes:")
            for name, prop in properties.items():
                print(f"  {name}: {prop.shape}")
            return properties
        except Exception as e:
            print(f"Error in get_mixture_properties: {str(e)}")
            print(f"Field shape: {phi.shape}")
            raise
    
    @property
    def phase_names(self) -> List[str]:
        """相の名前のリストを取得"""
        return list(self.phases.keys())
    
    def __str__(self) -> str:
        """文字列表現"""
        result = "FluidProperties:\n"
        for name, phase in self.phases.items():
            result += f"  {name}:\n"
            result += f"    密度: {phase.density} kg/m³\n"
            result += f"    粘性係数: {phase.viscosity} Pa·s\n"
            if phase.surface_tension is not None:
                result += f"    表面張力係数: {phase.surface_tension} N/m\n"
        return result