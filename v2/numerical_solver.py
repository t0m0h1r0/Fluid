import numpy as np
import numba
from typing import Dict, Tuple

class NumericalSolver:
    """結合コンパクト差分法に基づく数値ソルバー"""
    def __init__(self, grid_shape: Tuple[int, int, int]):
        """ソルバーの初期化"""
        self.grid_shape = grid_shape
        self._setup_coefficients()

    def _setup_coefficients(self):
        """CCDスキームの係数設定"""
        # 8次精度の内部点係数
        self.a = -1/32    # 3階微分
        self.b = 4/35     # 2階微分
        self.c = 105/16   # 1階微分
        
        # 境界点の係数
        self.boundary_coeffs = {
            'first': {
                'a': [-1.5, 2.0, -0.5],     # 前方差分
                'b': [1.0, -2.0, 1.0]       # 2階中心差分
            },
            'last': {
                'a': [0.5, -2.0, 1.5],      # 後方差分
                'b': [1.0, -2.0, 1.0]       # 2階中心差分
            }
        }

    @staticmethod
    @numba.njit
    def _calc_derivatives(
        f: np.ndarray, 
        dx: float, 
        a: float, 
        b: float, 
        c: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1方向の微分計算"""
        n = f.shape[0]
        f1 = np.zeros_like(f)  # 1階微分
        f2 = np.zeros_like(f)  # 2階微分
        f3 = np.zeros_like(f)  # 3階微分
        
        # 内部点の微分
        for i in range(1, n-1):
            # 1階微分
            f1[i] = (c * (f[i+1] - f[i-1]) + 
                     b * (f[i+2] - f[i-2]) +
                     a * (f[i+3] - f[i-3])) / (2*dx)
            
            # 2階微分
            f2[i] = (2*b * (f[i+1] + f[i-1] - 2*f[i]) +
                     a * (f[i+2] + f[i-2] - 2*f[i])) / (dx*dx)
            
            # 3階微分
            f3[i] = (c * (f[i+1] - f[i-1]) +
                     a * (f[i+2] - f[i-2])) / (dx*dx*dx)
        
        return f1, f2, f3

    def apply_derivative(
        self, 
        f: np.ndarray, 
        axis: int, 
        dx: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """指定軸方向の微分を計算"""
        if axis == 0:
            return self._calc_derivatives(f, dx, self.a, self.b, self.c)
        elif axis == 1:
            return self._calc_derivatives(f.transpose(1,0,2), dx, self.a, self.b, self.c)
        else:
            return self._calc_derivatives(f.transpose(2,0,1), dx, self.a, self.b, self.c)

    def divergence(
        self, 
        u: np.ndarray, 
        v: np.ndarray, 
        w: np.ndarray, 
        dx: float, 
        dy: float, 
        dz: float
    ) -> np.ndarray:
        """速度場の発散を計算"""
        ux, _, _ = self.apply_derivative(u, 0, dx)
        vy, _, _ = self.apply_derivative(v, 1, dy)
        wz, _, _ = self.apply_derivative(w, 2, dz)
        return ux + vy + wz

    def gradient(
        self, 
        f: np.ndarray, 
        dx: float, 
        dy: float, 
        dz: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """スカラー場の勾配を計算"""
        fx, _, _ = self.apply_derivative(f, 0, dx)
        fy, _, _ = self.apply_derivative(f, 1, dy)
        fz, _, _ = self.apply_derivative(f, 2, dz)
        return fx, fy, fz

    def laplacian(
        self, 
        f: np.ndarray, 
        dx: float, 
        dy: float, 
        dz: float
    ) -> np.ndarray:
        """スカラー場のラプラシアンを計算"""
        _, fxx, _ = self.apply_derivative(f, 0, dx)
        _, fyy, _ = self.apply_derivative(f, 1, dy)
        _, fzz, _ = self.apply_derivative(f, 2, dz)
        return fxx + fyy + fzz

    def solve_poisson(
        self, 
        b: np.ndarray, 
        dx: float, 
        dy: float, 
        dz: float, 
        max_iter: int = 1000, 
        tolerance: float = 1e-6
    ) -> np.ndarray:
        """ポアソン方程式の反復解法"""
        x = np.zeros_like(b)
        
        for _ in range(max_iter):
            x_old = x.copy()
            
            for i in range(1, b.shape[0]-1):
                for j in range(1, b.shape[1]-1):
                    for k in range(1, b.shape[2]-1):
                        x[i,j,k] = (
                            (x_old[i+1,j,k] + x_old[i-1,j,k]) / (dx*dx) +
                            (x_old[i,j+1,k] + x_old[i,j-1,k]) / (dy*dy) +
                            (x_old[i,j,k+1] + x_old[i,j,k-1]) / (dz*dz) -
                            b[i,j,k]
                        ) / (2/(dx*dx) + 2/(dy*dy) + 2/(dz*dz))
            
            # 収束判定
            if np.max(np.abs(x - x_old)) < tolerance:
                break
        
        return x
