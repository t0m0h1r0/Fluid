# physics/navier_stokes.py
import numpy as np
from typing import Tuple, List
from core.scheme import DifferenceScheme
from core.boundary import DirectionalBC
from .fluid_properties import MultiPhaseProperties
from numerics.poisson_solver.multigrid_poisson_solver import MultigridPoissonSolver

class NavierStokesSolver:
    def __init__(self,
                 scheme: DifferenceScheme,
                 boundary_conditions: DirectionalBC,
                 poisson_solver: MultigridPoissonSolver,
                 fluid_properties: MultiPhaseProperties):
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions
        self.poisson_solver = poisson_solver
        self.fluid_properties = fluid_properties
        self.gravity = 9.81

    def compute_timestep(self,
                        velocity: List[np.ndarray],
                        density: np.ndarray,
                        viscosity: np.ndarray,
                        dx: float) -> float:
        """CFL条件に基づく適応的時間刻み幅の計算"""
        # 対流項によるCFL条件
        max_velocity = max(np.max(np.abs(v)) for v in velocity)
        dt_convection = 0.5 * dx / (max_velocity + 1e-10)
        
        # 粘性項による制限
        max_viscosity = np.max(viscosity)
        min_density = np.max(density)
        dt_diffusion = 0.25 * dx**2 * min_density / (max_viscosity + 1e-10)
        
        # 表面張力による制限（Phase-Fieldモデルと結合時に使用）
        surface_tension = 0.07  # N/m
        dt_surface = np.sqrt(min_density * dx**3 / (2 * np.pi * surface_tension + 1e-10))
        
        return min(dt_convection, dt_diffusion, dt_surface)

    def compute_convection(self, 
                          velocity: List[np.ndarray],
                          density: np.ndarray,
                          axis: int) -> np.ndarray:
        """WENOスキームを用いた移流項の計算"""
        result = np.zeros_like(velocity[axis])
        
        for i in range(len(velocity)):
            # 風上差分の方向を決定
            upwind = velocity[i] < 0
            
            # WENO5による風上差分
            flux_plus = self._weno5(velocity[axis], axis, False)
            flux_minus = self._weno5(velocity[axis], axis, True)
            
            # フラックスの合成
            flux = np.where(upwind, flux_minus, flux_plus)
            result += velocity[i] * flux
        
        return result

    def _weno5(self, field: np.ndarray, axis: int, is_negative: bool) -> np.ndarray:
        """5次精度WENOスキームの実装"""
        # WENOの重み係数
        epsilon = 1e-6
        gamma0, gamma1, gamma2 = 0.1, 0.6, 0.3
        
        # シフトしたインデックスを準備
        if is_negative:
            v1 = np.roll(field, 2, axis=axis)
            v2 = np.roll(field, 1, axis=axis)
            v3 = field
            v4 = np.roll(field, -1, axis=axis)
            v5 = np.roll(field, -2, axis=axis)
        else:
            v1 = np.roll(field, -2, axis=axis)
            v2 = np.roll(field, -1, axis=axis)
            v3 = field
            v4 = np.roll(field, 1, axis=axis)
            v5 = np.roll(field, 2, axis=axis)
        
        # 3つの候補ステンシル
        s0 = (13/12) * (v1 - 2*v2 + v3)**2 + (1/4) * (v1 - 4*v2 + 3*v3)**2
        s1 = (13/12) * (v2 - 2*v3 + v4)**2 + (1/4) * (v2 - v4)**2
        s2 = (13/12) * (v3 - 2*v4 + v5)**2 + (1/4) * (3*v3 - 4*v4 + v5)**2
        
        # 非線形重み
        alpha0 = gamma0 / (epsilon + s0)**2
        alpha1 = gamma1 / (epsilon + s1)**2
        alpha2 = gamma2 / (epsilon + s2)**2
        omega = np.array([alpha0, alpha1, alpha2])
        omega /= np.sum(omega, axis=0)
        
        # 各ステンシルでの補間値
        p0 = (1/6) * (2*v1 - 7*v2 + 11*v3)
        p1 = (1/6) * (-v2 + 5*v3 + 2*v4)
        p2 = (1/6) * (2*v3 + 5*v4 - v5)
        
        return omega[0]*p0 + omega[1]*p1 + omega[2]*p2

    def compute_diffusion(self,
                         velocity: List[np.ndarray],
                         density: np.ndarray,
                         viscosity: np.ndarray,
                         axis: int) -> np.ndarray:
        """空間変化する粘性を考慮した拡散項の計算"""
        # 粘性の勾配を計算
        grad_mu = [self.compute_gradient(viscosity, i) for i in range(3)]
        
        # 速度の勾配テンソルを計算
        grad_u = [self.compute_gradient(velocity[axis], i) for i in range(3)]
        
        # ∇・(μ(∇u + ∇u^T))の計算
        result = np.zeros_like(velocity[axis])
        
        # 主要項 μ∇²u
        result += viscosity * self.compute_laplacian(velocity[axis])
        
        # 追加項 (∇μ・∇)u
        for i in range(3):
            result += grad_mu[i] * grad_u[i]
        
        return result / density

    def pressure_projection(self, 
                          velocity: List[np.ndarray],
                          density: np.ndarray,
                          dt: float) -> Tuple[List[np.ndarray], np.ndarray]:
        """圧力投影法による速度場の補正"""
        # 速度の発散を計算
        div_u = np.zeros_like(density)
        for i, v in enumerate(velocity):
            div_u += self.compute_gradient(v, i)
        
        # 圧力ポアソン方程式のソース項
        rhs = density * div_u / dt
        
        # マルチグリッド法で圧力を解く
        pressure = self.poisson_solver.solve(rhs)
        
        # 速度場の補正
        velocity_corrected = []
        for i, v in enumerate(velocity):
            grad_p = self.compute_gradient(pressure, i)
            v_new = v - dt * grad_p / density
            velocity_corrected.append(v_new)
        
        return velocity_corrected, pressure

    def runge_kutta4(self,
                    velocity: List[np.ndarray],
                    density: np.ndarray,
                    viscosity: np.ndarray,
                    dt: float) -> List[np.ndarray]:
        """4次のルンゲクッタ法による時間積分"""
        def compute_rhs(v: List[np.ndarray], rho: np.ndarray) -> List[np.ndarray]:
            """右辺項の計算"""
            rhs = []
            for axis in range(len(v)):
                # 移流項
                advection = self.compute_advection(v, axis)
                
                # 粘性項
                diffusion = self.compute_laplacian(v[axis])
                
                # 重力項（z方向のみ）
                gravity = np.zeros_like(v[axis])
                if axis == 2:  # z方向
                    # 密度の正規化を安全に行う
                    rho_min, rho_max = rho.min(), rho.max()
                    if np.isclose(rho_min, rho_max):
                        # 密度が一様な場合
                        gravity = -self.gravity * np.ones_like(v[axis])
                    else:
                        # 密度に応じた重力項
                        gravity = -self.gravity * (rho - rho_min) / (rho_max - rho_min)
                
                # 全項の合計
                total = (-advection + viscosity * diffusion / rho + gravity)
                rhs.append(total)
            
            return rhs

        # RK4のステージ
        k1 = compute_rhs(velocity, density)
        k1 = [dt * k for k in k1]
        
        v2 = [v + 0.5*k for v, k in zip(velocity, k1)]
        k2 = compute_rhs(v2, density)
        k2 = [dt * k for k in k2]
        
        v3 = [v + 0.5*k for v, k in zip(velocity, k2)]
        k3 = compute_rhs(v3, density)
        k3 = [dt * k for k in k3]
        
        v4 = [v + k for v, k in zip(velocity, k3)]
        k4 = compute_rhs(v4, density)
        k4 = [dt * k for k in k4]
        
        # 最終的な速度場の更新
        new_velocity = []
        for i in range(len(velocity)):
            v_new = velocity[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.0
            new_velocity.append(v_new)
        
        return new_velocity