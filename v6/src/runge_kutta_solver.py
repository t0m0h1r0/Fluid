# solvers/time_integration.py
import numpy as np
from typing import List, Optional
from solvers.interfaces import TimeIntegrationSolver
from physics.navier_stokes import NavierStokesSolver

class RungeKutta4Solver(TimeIntegrationSolver):
    def __init__(self, ns_solver: NavierStokesSolver):
        """
        4次のルンゲクッタ法による時間積分ソルバー

        Args:
            ns_solver: Navier-Stokesソルバー（各項の計算に使用）
        """
        self.ns_solver = ns_solver

    def solve(self,
              velocity: List[np.ndarray],
              density: np.ndarray,
              viscosity: np.ndarray,
              dt: float,
              external_forces: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        4次のルンゲクッタ法による時間積分

        Args:
            velocity: 現在の速度場
            density: 密度場
            viscosity: 粘性係数場
            dt: 時間刻み幅
            external_forces: 外部力（オプション）

        Returns:
            更新された速度場
        """
        def compute_rhs(v: List[np.ndarray], rho: np.ndarray) -> List[np.ndarray]:
            """右辺項の計算"""
            rhs = []
            for axis in range(len(v)):
                # 移流項
                advection = self.ns_solver.compute_convection(v, rho, axis)
                
                # 粘性項
                diffusion = self.ns_solver.compute_laplacian(v[axis])
                
                # 重力項（z方向のみ）
                gravity = np.zeros_like(v[axis])
                if axis == 2:  # z方向
                    # 密度の正規化を安全に行う
                    rho_min, rho_max = rho.min(), rho.max()
                    if np.isclose(rho_min, rho_max):
                        # 密度が一様な場合
                        gravity = -9.81 * np.ones_like(v[axis])
                    else:
                        # 密度に応じた重力項
                        gravity = -9.81 * (rho - rho_min) / (rho_max - rho_min)
                
                # 外部力の追加（オプション）
                external_force = (external_forces[axis] 
                                 if external_forces is not None 
                                 else np.zeros_like(v[axis]))
                
                # 全項の合計
                total = (-advection + viscosity * diffusion / rho + gravity + external_force)
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
