import numpy as np
from typing import Dict, Union, List
from .config import GridConfig, PhysicalParams, TimeConfig
from .boundary import BoundaryCondition

class Field:
    """スカラー場の基底クラス"""
    def __init__(self, data: np.ndarray):
        self.data = data
    
    def __array__(self) -> np.ndarray:
        return self.data
    
    def copy(self) -> 'Field':
        return Field(self.data.copy())

class VectorField:
    """ベクトル場"""
    def __init__(self, u: np.ndarray, v: np.ndarray, w: np.ndarray):
        self.u = Field(u)
        self.v = Field(v)
        self.w = Field(w)
    
    def __getitem__(self, key: Union[int, str]) -> np.ndarray:
        if key == 0 or key == 'u':
            return self.u.data
        elif key == 1 or key == 'v':
            return self.v.data
        elif key == 2 or key == 'w':
            return self.w.data
        raise KeyError(f"Invalid key: {key}")
    
    def apply_bc(self, bc: BoundaryCondition) -> None:
        """境界条件の適用"""
        self.u.data = bc.apply(self.u.data, 0)
        self.v.data = bc.apply(self.v.data, 1)
        self.w.data = bc.apply(self.w.data, 2)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """辞書形式に変換"""
        return {
            'u': self.u.data,
            'v': self.v.data,
            'w': self.w.data
        }
    
    def copy(self) -> 'VectorField':
        return VectorField(
            self.u.data.copy(),
            self.v.data.copy(),
            self.w.data.copy()
        )

class SimulationState:
    """シミュレーション全体の状態を管理"""
    def __init__(self, initial_fields: Dict[str, np.ndarray], 
                 grid: GridConfig, 
                 params: PhysicalParams):
        self.grid = grid
        self.params = params
        
        # フィールドの初期化
        self.rho = initial_fields['rho']
        self.vel = VectorField(
            initial_fields['u'],
            initial_fields['v'],
            initial_fields['w']
        )
        self.p = initial_fields['p']
        
        # 履歴の初期化
        self.history: Dict[str, List[np.ndarray]] = {
            'rho': [self.rho.copy()],
            'u': [self.vel.u.copy()],
            'v': [self.vel.v.copy()],
            'w': [self.vel.w.copy()],
            'p': [self.p.copy()]
        }
    
    def update_fields(self, new_fields: Dict[str, np.ndarray], save: bool = True):
        """フィールドの更新"""
        self.rho = new_fields['rho']
        self.vel = VectorField(
            new_fields['u'],
            new_fields['v'],
            new_fields['w']
        )
        self.p = new_fields['p']
        
        if save:
            self.history['rho'].append(self.rho.copy())
            self.history['u'].append(self.vel.u.copy())
            self.history['v'].append(self.vel.v.copy())
            self.history['w'].append(self.vel.w.copy())
            self.history['p'].append(self.p.copy())
    
    def get_params_dict(self, time_config: TimeConfig, start_step: int = 0) -> Dict:
        """パラメータの辞書化"""
        return {
            'dt': time_config.dt,
            'nx': self.grid.nx,
            'ny': self.grid.ny,
            'nz': self.grid.nz,
            'Lx': self.grid.Lx,
            'Ly': self.grid.Ly,
            'Lz': self.grid.Lz,
            'gravity': self.params.gravity,
            'step': start_step
        }
