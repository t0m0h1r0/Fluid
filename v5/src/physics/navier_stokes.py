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
    
    def compute_gradient(self, field: np.ndarray, axis: int) -> np.ndarray:
        """指定方向の勾配を計算"""
        bc = self.boundary_conditions.get_condition(axis)
        grad = np.zeros_like(field)

        # 境界条件を適用して配列を拡張
        padded = np.pad(field, pad_width=1, mode='edge')
        
        # 中心差分
        slices_center = [slice(1, -1)] * field.ndim
        slices_forward = [slice(1, -1)] * field.ndim
        slices_backward = [slice(1, -1)] * field.ndim
        
        slices_forward[axis] = slice(2, None)
        slices_backward[axis] = slice(None, -2)
        
        grad = (padded[tuple(slices_forward)] - padded[tuple(slices_backward)]) / 2.0
        
        # 境界条件の適用
        return bc.apply_to_field(grad)

    def compute_advection(self, u: List[np.ndarray], axis: int) -> np.ndarray:
        """移流項の計算"""
        result = np.zeros_like(u[axis])
        
        for i in range(len(u)):
            grad = self.compute_gradient(u[axis], i)
            result += u[i] * grad
            
        return result

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """ラプラシアンを計算"""
        laplacian = np.zeros_like(field)
        
        # 各方向について2階微分を計算
        for axis in range(field.ndim):
            bc = self.boundary_conditions.get_condition(axis)
            padded = np.pad(field, pad_width=1, mode='edge')
            
            # スライスの準備
            slices_center = [slice(1, -1)] * field.ndim
            slices_forward = [slice(1, -1)] * field.ndim
            slices_backward = [slice(1, -1)] * field.ndim
            
            slices_forward[axis] = slice(2, None)
            slices_backward[axis] = slice(None, -2)
            
            # 2階中心差分
            laplacian += (padded[tuple(slices_forward)] - 2*padded[tuple(slices_center)] + 
                         padded[tuple(slices_backward)])
        
        # 境界条件の適用
        for axis in range(field.ndim):
            bc = self.boundary_conditions.get_condition(axis)
            laplacian = bc.apply_to_field(laplacian)
        
        return laplacian

    def pressure_projection(self, 
                          velocity: List[np.ndarray],
                          density: np.ndarray,
                          dt: float) -> Tuple[List[np.ndarray], np.ndarray]:
        """圧力投影法による速度場の補正"""
        # 速度の発散を計算
        div_u = np.zeros_like(density)
        for i, v in enumerate(velocity):
            div_u += self.compute_gradient(v, i)
        
        # 圧力ポアソン方程式を解く
        rhs = density * div_u / dt
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
                    gravity = -self.gravity * (rho - rho.min()) / (rho.max() - rho.min())
                
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