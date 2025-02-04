import numpy as np
from typing import Tuple, Dict
from .config import GridConfig, PhysicalParams

class InitialConditionBase:
    """初期条件の基底クラス"""
    def __init__(self, params: PhysicalParams):
        self.params = params
    
    def heaviside(self, x: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
        """スムーズ化されたヘヴィサイド関数"""
        return 0.5 * (1 + np.tanh(x/epsilon))

class TwoLayerFluidIC(InitialConditionBase):
    """二層流体の初期条件"""
    def __init__(self, 
                 params: PhysicalParams,
                 bubble_center: Tuple[float, float, float] = (0.5, 0.5, 0.2),
                 bubble_radius: float = 0.1,
                 interface_height: float = 1.8):
        super().__init__(params)
        self.bubble_center = bubble_center
        self.bubble_radius = bubble_radius
        self.interface_height = interface_height
        
    def initialize(self, grid: GridConfig) -> Dict[str, np.ndarray]:
        """初期場の生成"""
        # グリッド点の生成
        X, Y, Z = grid.get_grid_points()
        
        # 気泡までの距離を計算
        distance = np.sqrt(
            (X - self.bubble_center[0])**2 + 
            (Y - self.bubble_center[1])**2 + 
            (Z - self.bubble_center[2])**2
        )
        
        # 密度場の初期化
        rho = self.params.rho_water * np.ones_like(X)
        
        # 上層の窒素
        rho = rho * (1 - self.heaviside(Z - self.interface_height))
        
        # 気泡の窒素
        rho = rho * (1 - self.heaviside(self.bubble_radius - distance))
        
        # 窒素の密度を加える
        rho += self.params.rho_nitrogen * self.heaviside(Z - self.interface_height)
        rho += self.params.rho_nitrogen * self.heaviside(self.bubble_radius - distance)
        
        # 速度場は初期静止
        u = np.zeros_like(X)
        v = np.zeros_like(X)
        w = np.zeros_like(X)
        
        # 圧力場は静水圧分布
        p = self.params.rho_water * self.params.gravity * (grid.Lz - Z)
        
        return {
            'rho': rho,
            'u': u,
            'v': v,
            'w': w,
            'p': p
        }
