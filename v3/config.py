import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class GridConfig:
    """格子設定"""
    nx: int
    ny: int
    nz: int
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 2.0
    
    @property
    def dx(self) -> float:
        return self.Lx / (self.nx - 1)
    
    @property
    def dy(self) -> float:
        return self.Ly / (self.ny - 1)
    
    @property
    def dz(self) -> float:
        return self.Lz / (self.nz - 1)
    
    def get_grid_points(self):
        """グリッド点の生成"""
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        z = np.linspace(0, self.Lz, self.nz)
        return np.meshgrid(x, y, z, indexing='ij')

@dataclass
class PhysicalParams:
    """物理パラメータ"""
    gravity: float = 9.81
    rho_water: float = 1000.0
    rho_nitrogen: float = 1.225
    dt: float = 0.001
    viscosity: float = 1.0e-6  # 動粘性係数
    surface_tension: float = 0.072  # 表面張力係数 [N/m]

@dataclass
class SolverParams:
    """ソルバーパラメータ"""
    max_iter: int = 50  # 最大反復回数
    tol: float = 1e-6   # 収束判定基準
    cfl: float = 0.5    # CFL数

@dataclass
class TimeConfig:
    """時間発展の設定"""
    dt: float
    max_steps: int
    save_interval: int = 10
