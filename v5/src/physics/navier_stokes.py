from typing import List, Tuple
import numpy as np

from core.scheme import DifferenceScheme, BoundaryCondition
from numerics.poisson_solver.abstract_poisson_solver import AbstractPoissonSolver
from .fluid_properties import MultiPhaseProperties

class NavierStokesSolver:
    def __init__(self,
                 scheme: DifferenceScheme,
                 boundary_conditions: List[BoundaryCondition],
                 poisson_solver: AbstractPoissonSolver,
                 fluid_properties: MultiPhaseProperties):
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions
        self.poisson_solver = poisson_solver
        self.fluid_properties = fluid_properties
    
    def compute_advection(self, u: List[np.ndarray], axis: int) -> np.ndarray:
        """移流項の計算"""
        result = np.zeros_like(u[axis])
        
        # 各速度成分の勾配計算
        gradients = [self._compute_gradient(u[i], i) for i in range(len(u))]
        
        # 移流項の計算
        for i, gradient in enumerate(gradients):
            result += u[i] * gradient
        
        return result
    
    def compute_diffusion(self, u: np.ndarray, viscosity: np.ndarray, axis: int) -> np.ndarray:
        """拡散項の計算"""
        # 2階微分（ラプラシアン）
        laplacian = np.zeros_like(u)
        
        # 各軸方向の2階微分
        for i in range(u.ndim):
            # 1次微分
            grad = self._compute_gradient(u, i)
            
            # 2次微分
            grad_grad = self._compute_gradient(grad, i)
            
            # ラプラシアンに加算
            laplacian += grad_grad
        
        return viscosity * laplacian
    
    def _compute_gradient(self, u: np.ndarray, axis: int) -> np.ndarray:
        """
        指定した軸方向の勾配計算
        
        Args:
            u (np.ndarray): 入力フィールド
            axis (int): 微分を計算する軸
        
        Returns:
            np.ndarray: 勾配フィールド
        """
        # 境界条件の取得
        bc = self.boundary_conditions.get_condition(axis)
        
        # 勾配計算
        gradient = np.zeros_like(u)
        
        # 各直交インデックスに対して勾配を計算
        for idx in self._get_orthogonal_indices(u.shape, axis):
            # インデックスに沿ったスライスの取得
            line = self._get_line(u, axis, idx)
            
            # 勾配の計算
            grad_line = self.scheme.apply(line, bc)
            
            # 結果の設定
            self._set_line(gradient, axis, idx, grad_line)
        
        return gradient
    
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
    
    def pressure_projection(self, 
                          velocity: List[np.ndarray],
                          density: np.ndarray,
                          dt: float) -> Tuple[List[np.ndarray], np.ndarray]:
        """圧力投影法"""
        # 速度の発散を計算
        div_u = np.zeros_like(density)
        for axis in range(len(velocity)):
            div_u += self._compute_gradient(velocity[axis], axis)
        
        # 圧力ポアソン方程式を解く
        rhs = density * div_u / dt
        pressure = self.poisson_solver.solve(rhs)
        
        # 速度場の補正
        corrected_velocity = []
        for axis in range(len(velocity)):
            # 圧力勾配の計算
            grad_p = self._compute_gradient(pressure, axis)
            
            # 速度補正
            v_corrected = velocity[axis] - dt * grad_p / density
            corrected_velocity.append(v_corrected)
        
        return corrected_velocity, pressure
    
    def runge_kutta4(self,
                     velocity: List[np.ndarray],
                     density: np.ndarray,
                     dt: float) -> List[np.ndarray]:
        """4次のルンゲクッタ法による時間発展"""
        def compute_rhs(v: List[np.ndarray]) -> List[np.ndarray]:
            rhs = []
            for axis in range(len(v)):
                # 移流項
                advection = self.compute_advection(v, axis)
                
                # 拡散項（粘性係数を考慮）
                viscosity = self.fluid_properties.get_viscosity(density)
                diffusion = self.compute_diffusion(v[axis], viscosity, axis)
                
                # 重力項（z方向のみ）
                gravity = np.zeros_like(v[axis])
                if axis == 2:  # z方向
                    gravity -= 9.81
                
                rhs.append(-advection + diffusion/density + gravity)
            return rhs
        
        # RK4のステージ
        k1 = compute_rhs(velocity)
        k1 = [dt * k for k in k1]
        
        v2 = [v + 0.5*k for v, k in zip(velocity, k1)]
        k2 = compute_rhs(v2)
        k2 = [dt * k for k in k2]
        
        v3 = [v + 0.5*k for v, k in zip(velocity, k2)]
        k3 = compute_rhs(v3)
        k3 = [dt * k for k in k3]
        
        v4 = [v + k for v, k in zip(velocity, k3)]
        k4 = compute_rhs(v4)
        k4 = [dt * k for k in k4]
        
        # 最終的な速度場の更新
        new_velocity = []
        for i in range(len(velocity)):
            v_new = velocity[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6
            new_velocity.append(v_new)
        
        return new_velocity