import numpy as np
from typing import Tuple, Dict
from .config import GridConfig, PhysicalParams, SolverParams
from .fields import VectorField

class CCDOperator:
    """結合コンパクト差分演算子"""
    def __init__(self, grid: GridConfig):
        self.grid = grid
        self.setup_coefficients()
        
    def setup_coefficients(self):
        """CCDスキームの係数設定"""
        # 内部点の係数 (8次精度)
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
        
    def apply_derivative(self, f: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """指定軸方向の微分を計算"""
        if axis == 0:
            return self._calc_derivatives(f, self.grid.dx)
        elif axis == 1:
            return self._calc_derivatives(f.transpose(1,0,2), self.grid.dy)
        else:
            return self._calc_derivatives(f.transpose(2,0,1), self.grid.dz)
            
    def _calc_derivatives(self, f: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1方向の微分計算"""
        n = f.shape[0]
        f1 = np.zeros_like(f)  # 1階微分
        f2 = np.zeros_like(f)  # 2階微分
        f3 = np.zeros_like(f)  # 3階微分
        
        # 内部点
        for i in range(1, n-1):
            # 1階微分
            f1[i] = (self.c * (f[i+1] - f[i-1]) + 
                    self.b * (f[i+2] - f[i-2]) +
                    self.a * (f[i+3] - f[i-3])) / (2*dx)
            
            # 2階微分
            f2[i] = (2*self.b * (f[i+1] + f[i-1] - 2*f[i]) +
                    self.a * (f[i+2] + f[i-2] - 2*f[i])) / (dx*dx)
            
            # 3階微分
            f3[i] = (self.c * (f[i+1] - f[i-1]) +
                    self.a * (f[i+2] - f[i-2])) / (dx*dx*dx)
        
        # 境界点
        bc = self.boundary_coeffs
        
        # 左境界
        f1[0] = sum(c*f[i] for i, c in enumerate(bc['first']['a'])) / dx
        f2[0] = sum(c*f[i] for i, c in enumerate(bc['first']['b'])) / (dx*dx)
        
        # 右境界
        f1[-1] = sum(c*f[-(i+1)] for i, c in enumerate(bc['last']['a'])) / dx
        f2[-1] = sum(c*f[-(i+1)] for i, c in enumerate(bc['last']['b'])) / (dx*dx)
        
        return f1, f2, f3
        
    def divergence(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """速度場の発散を計算"""
        ux, _, _ = self.apply_derivative(u, 0)
        vy, _, _ = self.apply_derivative(v, 1)
        wz, _, _ = self.apply_derivative(w, 2)
        return ux + vy + wz
        
    def gradient(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """スカラー場の勾配を計算"""
        fx, _, _ = self.apply_derivative(f, 0)
        fy, _, _ = self.apply_derivative(f, 1)
        fz, _, _ = self.apply_derivative(f, 2)
        return fx, fy, fz
        
    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """スカラー場のラプラシアンを計算"""
        _, fxx, _ = self.apply_derivative(f, 0)
        _, fyy, _ = self.apply_derivative(f, 1)
        _, fzz, _ = self.apply_derivative(f, 2)
        return fxx + fyy + fzz

class PressureSolver:
    """圧力解法クラス"""
    def __init__(self, grid: GridConfig, params: SolverParams):
        self.grid = grid
        self.max_iter = params.max_iter
        self.tol = params.tol
        self.ccd = CCDOperator(grid)
        
    def solve(self, velocity: VectorField, density: np.ndarray, dt: float) -> np.ndarray:
        """圧力のPoisson方程式を解く"""
        # 発散の計算
        div = self.ccd.divergence(velocity.u, velocity.v, velocity.w)
        
        # 圧力場の初期化
        p = np.zeros_like(div)
        
        # ガウス・ザイデル法による反復計算
        dx2 = self.grid.dx * self.grid.dx
        dy2 = self.grid.dy * self.grid.dy
        dz2 = self.grid.dz * self.grid.dz
        
        for _ in range(self.max_iter):
            p_old = p.copy()
            
            for i in range(1, self.grid.nx-1):
                for j in range(1, self.grid.ny-1):
                    for k in range(1, self.grid.nz-1):
                        p[i,j,k] = (
                            ((p_old[i+1,j,k] + p_old[i-1,j,k])/dx2 +
                             (p_old[i,j+1,k] + p_old[i,j-1,k])/dy2 +
                             (p_old[i,j,k+1] + p_old[i,j,k-1])/dz2 -
                             div[i,j,k]/dt) / 
                            (2/dx2 + 2/dy2 + 2/dz2)
                        )
            
            # 収束判定
            if np.max(np.abs(p - p_old)) < self.tol:
                break
        
        return p
        
    def correct_velocity(self, 
                        velocity: VectorField, 
                        pressure: np.ndarray,
                        density: np.ndarray,
                        dt: float) -> VectorField:
        """速度場の修正"""
        # 圧力勾配の計算
        px, py, pz = self.ccd.gradient(pressure)
        
        # 速度場の修正
        u_new = velocity.u - dt * px / density
        v_new = velocity.v - dt * py / density
        w_new = velocity.w - dt * pz / density
        
        return VectorField(u_new, v_new, w_new)

class TimeEvolver:
    """時間発展計算クラス"""
    def __init__(self,
                 grid: GridConfig,
                 physical_params: PhysicalParams,
                 ccd: CCDOperator,
                 pressure: PressureSolver):
        self.grid = grid
        self.params = physical_params
        self.ccd = ccd
        self.pressure = pressure
        
    def rk4_step(self, 
                 rho: np.ndarray,
                 vel: VectorField,
                 p: np.ndarray,
                 dt: float) -> Dict[str, np.ndarray]:
        """4次のルンゲクッタ法による時間発展"""
        def velocity_rhs(v: VectorField, rho: np.ndarray, p: np.ndarray):
            """速度場の右辺"""
            px, py, pz = self.ccd.gradient(p)
            du = -px/rho
            dv = -py/rho
            dw = -pz/rho - self.params.gravity
            return VectorField(du, dv, dw)
            
        # RK4の4段階計算
        k1 = velocity_rhs(vel, rho, p)
        
        vel_k2 = VectorField(
            vel.u + 0.5*dt*k1.u,
            vel.v + 0.5*dt*k1.v,
            vel.w + 0.5*dt*k1.w
        )
        p_k2 = self.pressure.solve(vel_k2, rho, dt)
        k2 = velocity_rhs(vel_k2, rho, p_k2)
        
        vel_k3 = VectorField(
            vel.u + 0.5*dt*k2.u,
            vel.v + 0.5*dt*k2.v,
            vel.w + 0.5*dt*k2.w
        )
        p_k3 = self.pressure.solve(vel_k3, rho, dt)
        k3 = velocity_rhs(vel_k3, rho, p_k3)
        
        vel_k4 = VectorField(
            vel.u + dt*k3.u,
            vel.v + dt*k3.v,
            vel.w + dt*k3.w
        )
        p_k4 = self.pressure.solve(vel_k4, rho, dt)
        k4 = velocity_rhs(vel_k4, rho, p_k4)
        
        # 速度場の更新
        u_new = vel.u + (dt/6)*(k1.u + 2*k2.u + 2*k3.u + k4.u)
        v_new = vel.v + (dt/6)*(k1.v + 2*k2.v + 2*k3.v + k4.v)
        w_new = vel.w + (dt/6)*(k1.w + 2*k2.w + 2*k3.w + k4.w)
        
        # 圧力場の最終更新
        vel_new = VectorField(u_new, v_new, w_new)
        p_new = self.pressure.solve(vel_new, rho, dt)
        
        return {
            'u': u_new,
            'v': v_new,
            'w': w_new,
            'p': p_new
        }
    
    def advect_density(self, rho: np.ndarray, vel: VectorField, dt: float) -> np.ndarray:
        """周期境界条件を考慮した密度場の移流"""
        rho_new = np.zeros_like(rho)
        
        for i in range(1, self.grid.nx-1):
            for j in range(1, self.grid.ny-1):
                for k in range(1, self.grid.nz-1):
                    # 特性曲線の追跡（周期境界条件を考慮）
                    x = (i - vel.u[i,j,k] * dt / self.grid.dx) % (self.grid.nx - 1)
                    y = (j - vel.v[i,j,k] * dt / self.grid.dy) % (self.grid.ny - 1)
                    z = (k - vel.w[i,j,k] * dt / self.grid.dz) % (self.grid.nz - 1)
                    
                    # 格子点の補間
                    x0, y0, z0 = int(x), int(y), int(z)
                    tx, ty, tz = x - x0, y - y0, z - z0
                    
                    # 三線形補間（周期境界条件を考慮）
                    c000 = rho[x0 % (self.grid.nx-1), y0 % (self.grid.ny-1), z0 % (self.grid.nz-1)]
                    c001 = rho[x0 % (self.grid.nx-1), y0 % (self.grid.ny-1), (z0+1) % (self.grid.nz-1)]
                    c010 = rho[x0 % (self.grid.nx-1), (y0+1) % (self.grid.ny-1), z0 % (self.grid.nz-1)]
                    c011 = rho[x0 % (self.grid.nx-1), (y0+1) % (self.grid.ny-1), (z0+1) % (self.grid.nz-1)]
                    c100 = rho[(x0+1) % (self.grid.nx-1), y0 % (self.grid.ny-1), z0 % (self.grid.nz-1)]
                    c101 = rho[(x0+1) % (self.grid.nx-1), y0 % (self.grid.ny-1), (z0+1) % (self.grid.nz-1)]
                    c110 = rho[(x0+1) % (self.grid.nx-1), (y0+1) % (self.grid.ny-1), z0 % (self.grid.nz-1)]
                    c111 = rho[(x0+1) % (self.grid.nx-1), (y0+1) % (self.grid.ny-1), (z0+1) % (self.grid.nz-1)]
                    
                    # 三線形補間
                    rho_new[i,j,k] = (
                        c000 * (1-tx) * (1-ty) * (1-tz) +
                        c001 * (1-tx) * (1-ty) * tz +
                        c010 * (1-tx) * ty * (1-tz) +
                        c011 * (1-tx) * ty * tz +
                        c100 * tx * (1-ty) * (1-tz) +
                        c101 * tx * (1-ty) * tz +
                        c110 * tx * ty * (1