from typing import Dict, List, Optional
import numpy as np
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField
from core.solver.poisson.poisson import PoissonSolver
from .terms import AdvectionTerm, PressureTerm, ViscousTerm, ExternalForceTerm

class NavierStokesSolver:
    """Navier-Stokes方程式ソルバー"""
    def __init__(self, poisson_solver: PoissonSolver, *args, **kwargs):  # *args, **kwargsを追加
        self.poisson_solver = poisson_solver
        self.terms = [
            AdvectionTerm(),
            PressureTerm(),
            ViscousTerm(),
            ExternalForceTerm()
        ]

    def compute_right_hand_side(self, fields: Dict[str, ScalarField | VectorField]) -> List[np.ndarray]:
        """NS方程式の右辺を計算"""
        rhs = [np.zeros_like(fields['velocity'].data[0]) for _ in range(3)]
        
        # 各項の寄与を計算
        for term in self.terms:
            contribution = term.compute(fields)
            for i in range(3):
                rhs[i] += contribution[i]
                
        return rhs

    def pressure_correction(self, 
                          velocity: VectorField,
                          density: ScalarField,
                          dt: float) -> tuple[VectorField, ScalarField]:
        """圧力補正ステップ"""
        # 圧力場の初期化
        pressure = ScalarField(velocity.metadata)
        
        # 速度の発散を計算
        div_v = np.zeros_like(density.data)
        for i, v in enumerate(velocity.data):
            div_v += np.gradient(v, velocity.dx[i], axis=i)
        
        # 圧力ポアソン方程式の右辺
        rhs = ScalarField(velocity.metadata)
        rhs.data = density.data * div_v / dt
        
        # 圧力を解く
        pressure = self.poisson_solver.solve(rhs)
        
        # 速度場の補正
        corrected_velocity = VectorField(velocity.metadata)
        corrected_data = []
        for i, v in enumerate(velocity.data):
            grad_p = np.gradient(pressure.data, velocity.dx[i], axis=i)
            v_new = v - dt * grad_p / density.data
            corrected_data.append(v_new)
        corrected_velocity.data = corrected_data
        
        return corrected_velocity, pressure

    def check_cfl_condition(self, 
                          velocity: VectorField,
                          dt: float) -> bool:
        """CFL条件のチェック"""
        max_velocity = max(np.max(np.abs(v)) for v in velocity.data)
        dx_min = min(velocity.dx)
        cfl = dt * max_velocity / dx_min
        return cfl <= 1.0

    def compute_adaptive_timestep(self, 
                                velocity: VectorField,
                                viscosity: ScalarField,
                                density: ScalarField,
                                cfl_target: float = 0.5) -> float:
        """適応的な時間刻み幅の計算"""
        # 対流による制限
        max_velocity = max(np.max(np.abs(v)) for v in velocity.data)
        dx_min = min(velocity.dx)
        dt_convection = cfl_target * dx_min / (max_velocity + 1e-10)
        
        # 粘性による制限
        max_viscosity = np.max(viscosity.data)
        min_density = np.min(density.data)
        dt_viscous = 0.25 * dx_min**2 * min_density / (max_viscosity + 1e-10)
        
        return min(dt_convection, dt_viscous)