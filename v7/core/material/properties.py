import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from core.field.scalar_field import ScalarField
from core.field.metadata import FieldMetadata

@dataclass
class FluidProperties:
    """流体の物性値"""
    name: str
    density: float
    viscosity: float
    surface_tension: Optional[float] = None
    specific_heat: Optional[float] = None
    thermal_conductivity: Optional[float] = None

class MaterialManager:
    """物性値管理クラス"""
    def __init__(self):
        self.fluids: Dict[str, FluidProperties] = {}
        self.phase_field: Optional[ScalarField] = None
        self.density_field: Optional[ScalarField] = None
        self.viscosity_field: Optional[ScalarField] = None
    
    def add_fluid(self, name: str, properties: FluidProperties):
        """流体の追加"""
        self.fluids[name] = properties

    def initialize_phase_field(self, phase_field: ScalarField):
        """相場の初期化"""
        self.phase_field = phase_field
        
        # 密度場と粘性場も初期化
        self.density_field = ScalarField(FieldMetadata(
            name='density',
            unit='kg/m³',
            domain_size=phase_field.metadata.domain_size,
            resolution=phase_field.metadata.resolution
        ))
        
        self.viscosity_field = ScalarField(FieldMetadata(
            name='viscosity',
            unit='Pa·s',
            domain_size=phase_field.metadata.domain_size,
            resolution=phase_field.metadata.resolution
        ))
        
        # 初期値の計算
        self.update_properties(phase_field)

    def update_properties(self, phase_field: ScalarField):
        """物性値の更新"""
        if len(self.fluids) != 2:
            raise ValueError("現在は2相のみサポートしています")
        
        fluid1, fluid2 = list(self.fluids.values())
        
        # ヘビサイド関数による重み付け
        H = self._heaviside(phase_field.data)
        
        # 密度の更新
        self.density_field.data = (
            fluid1.density * H + fluid2.density * (1 - H)
        )
        
        # 粘性の更新（調和平均）
        self.viscosity_field.data = 1.0 / (
            H / fluid1.viscosity + (1 - H) / fluid2.viscosity
        )

    def get_density_field(self) -> ScalarField:
        """密度場の取得"""
        if self.density_field is None:
            raise RuntimeError("密度場が初期化されていません")
        return self.density_field

    def get_viscosity_field(self) -> ScalarField:
        """粘性場の取得"""
        if self.viscosity_field is None:
            raise RuntimeError("粘性場が初期化されていません")
        return self.viscosity_field

    def _heaviside(self, phi: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """スムーズなヘビサイド関数"""
        return 0.5 * (1.0 + np.tanh(phi / epsilon))