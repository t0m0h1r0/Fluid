import numpy as np
from typing import Tuple, List
from core.scheme import DifferenceScheme
from core.boundary import DirectionalBC
from .fluid_properties import MultiPhaseProperties
from numerics.poisson_solver.abstract_poisson_solver import AbstractPoissonSolver as PoissonSolver

class NavierStokesSolver:
    def __init__(self,
                 scheme: DifferenceScheme,
                 boundary_conditions: DirectionalBC,
                 poisson_solver: PoissonSolver,
                 fluid_properties: MultiPhaseProperties):
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions
        self.poisson_solver = poisson_solver
        self.fluid_properties = fluid_properties
        self.gravity = 9.81
        
    def compute_advection(self, u: List[np.ndarray], axis: int) -> np.ndarray:
        """移流項の計算
        Args:
            u: 速度場のリスト [u, v, w]
            axis: 計算する方向（0:x, 1:y, 2:z）
        Returns:
            移流項 (u・∇)u
        """
        result = np.zeros_like(u[axis])
        for i in range(len(u)):
            bc = self.boundary_conditions.get_condition(i)
            gradient = self.compute_gradient(u[axis], i)
            result += u[i] * gradient
        return result
    
    def compute_diffusion(self, u: np.ndarray, viscosity: np.ndarray, axis: int) -> np.ndarray:
        """粘性拡散項の計算
        Args:
            u: 速度場
            viscosity: 粘性係数場
            axis: 計算する方向（0:x, 1:y, 2:z）
        Returns:
            拡散項 ν∇²u
        """
        bc = self.boundary_conditions.get_condition(axis)
        laplacian = np.zeros_like(u)
        for i in range(u.ndim):
            for idx in self._get_orthogonal_indices(u.shape, i):
                line = self._get_line(u, i, idx)
                d2_line = self.scheme.apply(line, bc)
                self._set_line(laplacian, i, idx, d2_line)
        return viscosity * laplacian
    
    def compute_gradient(self, u: np.ndarray, axis: int) -> np.ndarray:
        """勾配の計算
        Args:
            u: スカラー場
            axis: 計算する方向（0:x, 1:y, 2:z）
        Returns:
            勾配 ∂u/∂x_i
        """
        bc = self.boundary_conditions.get_condition(axis)
        result = np.zeros_like(u)
        for idx in self._get_orthogonal_indices(u.shape, axis):
            line = self._get_line(u, axis, idx)
            grad_line = self.scheme.apply(line, bc)
            self._set_line(result, axis, idx, grad_line)
        return result
    
    def pressure_projection(self, 
                          velocity: List[np.ndarray],
                          density: np.ndarray,
                          dt: float) -> Tuple[List[np.ndarray], np.ndarray]:
        """圧力投影法
        Args:
            velocity: 速度場のリスト [u, v, w]
            density: 密度場
            dt: 時間刻み幅
        Returns:
            補正後の速度場と圧力場
        """
        # 速度の発散を計算
        div_u = np.zeros_like(density)
        for axis in range(len(velocity)):
            bc = self.boundary_conditions.get_condition(axis)
            div_u += self.compute_gradient(velocity[axis], axis)
        
        # 圧力ポアソン方程式を解く
        rhs = density * div_u / dt
        pressure = self.poisson_solver.solve(rhs)
        
        # 速度場の補正
        corrected_velocity = []
        for axis in range(len(velocity)):
            grad_p = self.compute_gradient(pressure, axis)
            v_corrected = velocity[axis] - dt * grad_p / density
            corrected_velocity.append(v_corrected)
        
        return corrected_velocity, pressure
    
    def runge_kutta4(self,
                    velocity: List[np.ndarray],
                    density: np.ndarray,
                    viscosity: np.ndarray,
                    dt: float) -> List[np.ndarray]:
        """4次のルンゲクッタ法による時間発展
        Args:
            velocity: 速度場のリスト [u, v, w]
            density: 密度場
            viscosity: 粘性係数場
            dt: 時間刻み幅
        Returns:
            更新された速度場のリスト
        """
        def compute_rhs(v: List[np.ndarray], rho: np.ndarray, nu: np.ndarray) -> List[np.ndarray]:
            """右辺の計算
            Args:
                v: 速度場のリスト
                rho: 密度場
                nu: 粘性係数場
            Returns:
                各方向の右辺項のリスト
            """
            rhs = []
            for axis in range(len(v)):
                # 移流項
                advection = self.compute_advection(v, axis)
                
                # 粘性拡散項
                diffusion = self.compute_diffusion(v[axis], nu, axis)
                
                # 重力項（z方向のみ）
                gravity = np.zeros_like(v[axis])
                if axis == 2:  # z方向
                    gravity = -self.gravity * (rho - rho.min()) / (rho.max() - rho.min())
                
                # 全項の合計
                total = (-advection + diffusion/rho + gravity)
                rhs.append(total)
            
            return rhs

        # RK4のステージ
        k1 = compute_rhs(velocity, density, viscosity)
        k1 = [dt * k for k in k1]
        
        v2 = [v + 0.5*k for v, k in zip(velocity, k1)]
        k2 = compute_rhs(v2, density, viscosity)
        k2 = [dt * k for k in k2]
        
        v3 = [v + 0.5*k for v, k in zip(velocity, k2)]
        k3 = compute_rhs(v3, density, viscosity)
        k3 = [dt * k for k in k3]
        
        v4 = [v + k for v, k in zip(velocity, k3)]
        k4 = compute_rhs(v4, density, viscosity)
        k4 = [dt * k for k in k4]
        
        # 最終的な速度場の更新
        new_velocity = []
        for i in range(len(velocity)):
            v_new = velocity[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6
            new_velocity.append(v_new)
        
        return new_velocity
    
    def _get_orthogonal_indices(self, shape: Tuple[int, ...], axis: int):
        """指定された軸に直交する全インデックスの組み合わせを生成"""
        ranges = [range(s) for i, s in enumerate(shape) if i != axis]
        return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(len(ranges), -1).T
    
    def _get_line(self, array: np.ndarray, axis: int, idx) -> np.ndarray:
        """指定された軸に沿ってラインを抽出"""
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        return array[tuple(idx_list)]
    
    def _set_line(self, array: np.ndarray, axis: int, idx, values: np.ndarray):
        """指定された軸に沿ってラインを設定"""
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        array[tuple(idx_list)] = values