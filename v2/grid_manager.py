from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import numba

@dataclass
class GridConfig:
    """グリッド設定の管理"""
    nx: int  # x方向のグリッド数
    ny: int  # y方向のグリッド数
    nz: int  # z方向のグリッド数
    
    Lx: float = 1.0  # x方向の計算領域長さ
    Ly: float = 1.0  # y方向の計算領域長さ
    Lz: float = 2.0  # z方向の計算領域長さ

    def __post_init__(self):
        """グリッド設定のバリデーション"""
        if any(dim <= 0 for dim in [self.nx, self.ny, self.nz, 
                                    self.Lx, self.Ly, self.Lz]):
            raise ValueError("グリッドパラメータは正の値である必要があります")

    @property
    def dx(self) -> float:
        """x方向のグリッド間隔"""
        return self.Lx / (self.nx - 1)

    @property
    def dy(self) -> float:
        """y方向のグリッド間隔"""
        return self.Ly / (self.ny - 1)

    @property
    def dz(self) -> float:
        """z方向のグリッド間隔"""
        return self.Lz / (self.nz - 1)

class GridManager:
    """グリッド生成と操作のための高性能クラス"""
    def __init__(self, config: GridConfig):
        self.config = config
        self._initialize_grid()

    def _initialize_grid(self):
        """グリッド点の生成"""
        self.x = np.linspace(0, self.config.Lx, self.config.nx)
        self.y = np.linspace(0, self.config.Ly, self.config.ny)
        self.z = np.linspace(0, self.config.Lz, self.config.nz)
        
        # メッシュグリッドの生成（高速化のためインデックス指定）
        self.X, self.Y, self.Z = np.meshgrid(
            self.x, self.y, self.z, indexing='ij'
        )

    @numba.njit
    def interpolate_trilinear(
        self, 
        field: np.ndarray, 
        x: float, 
        y: float, 
        z: float
    ) -> float:
        """高速三線形補間"""
        # インデックスの計算
        x0 = int(x)
        y0 = int(y)
        z0 = int(z)
        
        # 補間係数
        tx = x - x0
        ty = y - y0
        tz = z - z0
        
        # 範囲制限
        x0 = max(0, min(x0, field.shape[0] - 2))
        y0 = max(0, min(y0, field.shape[1] - 2))
        z0 = max(0, min(z0, field.shape[2] - 2))
        
        # 三線形補間
        c000 = field[x0, y0, z0]
        c100 = field[x0+1, y0, z0]
        c010 = field[x0, y0+1, z0]
        c110 = field[x0+1, y0+1, z0]
        c001 = field[x0, y0, z0+1]
        c101 = field[x0+1, y0, z0+1]
        c011 = field[x0, y0+1, z0+1]
        c111 = field[x0+1, y0+1, z0+1]
        
        return (
            c000 * (1-tx)*(1-ty)*(1-tz) +
            c100 * tx*(1-ty)*(1-tz) +
            c010 * (1-tx)*ty*(1-tz) +
            c110 * tx*ty*(1-tz) +
            c001 * (1-tx)*(1-ty)*tz +
            c101 * tx*(1-ty)*tz +
            c011 * (1-tx)*ty*tz +
            c111 * tx*ty*tz
        )

    def get_grid_indices(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """物理座標からグリッドインデックスへの変換"""
        ix = np.searchsorted(self.x, x) - 1
        iy = np.searchsorted(self.y, y) - 1
        iz = np.searchsorted(self.z, z) - 1
        return (
            max(0, min(ix, self.config.nx - 2)),
            max(0, min(iy, self.config.ny - 2)),
            max(0, min(iz, self.config.nz - 2))
        )

    def generate_volume_fractions(self, interface_function):
        """界面の体積分率計算"""
        volume_fractions = np.zeros_like(self.X, dtype=float)
        
        for i in range(self.config.nx):
            for j in range(self.config.ny):
                for k in range(self.config.nz):
                    volume_fractions[i,j,k] = interface_function(
                        self.X[i,j,k], 
                        self.Y[i,j,k], 
                        self.Z[i,j,k]
                    )
        
        return volume_fractions
