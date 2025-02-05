from typing import Dict, List
import numpy as np
from .terms import NavierStokesTerm, AdvectionTerm, PressureTerm, ViscousTerm, ExternalForceTerm
from ...field.scalar_field import ScalarField
from ...field.vector_field import VectorField

class NavierStokesSolver:
    """Navier-Stokes方程式ソルバー"""
    def __init__(self):
        self.terms: List[NavierStokesTerm] = [
            AdvectionTerm(),
            PressureTerm(),
            ViscousTerm(),
            ExternalForceTerm()
        ]
        self.fields: Dict[str, ScalarField | VectorField] = {}

    def set_fields(self, fields: Dict[str, ScalarField | VectorField]):
        required = {'velocity', 'pressure', 'density', 'viscosity'}
        if not all(name in fields for name in required):
            raise ValueError(f"必要なフィールドが不足しています: {required}")
        self.fields = fields

    def compute_right_hand_side(self) -> List[np.ndarray]:
        """NS方程式の右辺を計算"""
        rhs = [np.zeros_like(self.fields['velocity'].data[0]) for _ in range(3)]
        
        for term in self.terms:
            term_contribution = term.compute(self.fields)
            for i in range(3):
                rhs[i] += term_contribution[i]
        
        return rhs

    def check_cfl_condition(self, dt: float) -> bool:
        """CFL条件のチェック"""
        velocity = self.fields['velocity'].data
        dx = self.fields['velocity'].dx
        
        max_velocity = max(np.max(np.abs(v)) for v in velocity)
        cfl = dt * max_velocity / min(dx)
        
        return cfl <= 1.0

    def compute_adaptive_timestep(self, cfl_target: float = 0.5) -> float:
        """適応的な時間刻み幅の計算"""
        velocity = self.fields['velocity'].data
        dx = self.fields['velocity'].dx
        viscosity = self.fields['viscosity'].data
        density = self.fields['density'].data
        
        # 対流による制限
        max_velocity = max(np.max(np.abs(v)) for v in velocity)
        dt_convection = cfl_target * min(dx) / (max_velocity + 1e-10)
        
        # 粘性による制限
        max_viscosity = np.max(viscosity)
        min_density = np.min(density)
        dt_viscous = 0.25 * min(dx)**2 * min_density / (max_viscosity + 1e-10)
        
        return min(dt_convection, dt_viscous)