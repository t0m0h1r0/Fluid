from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import yaml
from pathlib import Path

@dataclass
class PhaseConfig:
    """相の設定"""
    name: str
    density: float
    viscosity: float
    surface_tension: Optional[float] = None

@dataclass
class GridConfig:
    """グリッドの設定"""
    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float

@dataclass
class TimeConfig:
    """時間発展の設定"""
    dt: float
    max_time: float
    save_interval: float
    cfl: float = 0.5

@dataclass
class SolverConfig:
    """ソルバーの設定"""
    poisson_solver: str = 'multigrid'
    poisson_tolerance: float = 1e-6
    poisson_max_iterations: int = 100
    time_integrator: str = 'rk4'
    convection_scheme: str = 'weno'

@dataclass
class BoundaryConfig:
    """境界条件の設定"""
    x: str
    y: str
    z: str

@dataclass
class InitialConditionConfig:
    """初期条件の設定"""
    type: str
    parameters: Dict[str, float]

class SimulationConfig:
    """シミュレーション全体の設定"""
    
    def __init__(self, config_file: str):
        """
        Args:
            config_file: 設定ファイルのパス
        """
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_file}")
        
        # 設定の読み込みと解析
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 各コンポーネントの設定を解析
        self.phases = self._parse_phases(config.get('phases', {}))
        self.grid = self._parse_grid(config.get('grid', {}))
        self.time = self._parse_time(config.get('time', {}))
        self.solver = self._parse_solver(config.get('solver', {}))
        self.boundary = self._parse_boundary(config.get('boundary', {}))
        self.initial = self._parse_initial(config.get('initial_condition', {}))
        
        # 出力設定
        self.output_dir = config.get('output', {}).get('directory', 'output')
        
        # 設定の検証
        self._validate()
    
    def _parse_phases(self, config: Dict) -> List[PhaseConfig]:
        """相の設定を解析"""
        phases = []
        for phase_data in config:
            phases.append(PhaseConfig(
                name=phase_data['name'],
                density=float(phase_data['density']),
                viscosity=float(phase_data['viscosity']),
                surface_tension=float(phase_data.get('surface_tension', 0.0))
            ))
        return phases
    
    def _parse_grid(self, config: Dict) -> GridConfig:
        """グリッドの設定を解析"""
        return GridConfig(
            nx=int(config.get('nx', 32)),
            ny=int(config.get('ny', 32)),
            nz=int(config.get('nz', 32)),
            lx=float(config.get('lx', 1.0)),
            ly=float(config.get('ly', 1.0)),
            lz=float(config.get('lz', 1.0))
        )
    
    def _parse_time(self, config: Dict) -> TimeConfig:
        """時間発展の設定を解析"""
        return TimeConfig(
            dt=float(config.get('dt', 0.001)),
            max_time=float(config.get('max_time', 1.0)),
            save_interval=float(config.get('save_interval', 0.1)),
            cfl=float(config.get('cfl', 0.5))
        )
    
    def _parse_solver(self, config: Dict) -> SolverConfig:
        """ソルバーの設定を解析"""
        return SolverConfig(
            poisson_solver=str(config.get('poisson_solver', 'multigrid')),
            poisson_tolerance=float(config.get('poisson_tolerance', 1e-6)),
            poisson_max_iterations=int(config.get('poisson_max_iterations', 100)),
            time_integrator=str(config.get('time_integrator', 'rk4')),
            convection_scheme=str(config.get('convection_scheme', 'weno'))
        )
    
    def _parse_boundary(self, config: Dict) -> BoundaryConfig:
        """境界条件の設定を解析"""
        return BoundaryConfig(
            x=str(config.get('x', 'periodic')),
            y=str(config.get('y', 'periodic')),
            z=str(config.get('z', 'neumann'))
        )
    
    def _parse_initial(self, config: Dict) -> InitialConditionConfig:
        """初期条件の設定を解析"""
        return InitialConditionConfig(
            type=str(config.get('type', 'two_layer')),
            parameters=config.get('parameters', {})
        )
    
    def _validate(self):
        """設定の妥当性を検証"""
        # グリッドサイズの検証
        if any(size <= 0 for size in [self.grid.nx, self.grid.ny, self.grid.nz]):
            raise ValueError("グリッドサイズは正の整数である必要があります")
        
        # 時間パラメータの検証
        if self.time.dt <= 0 or self.time.max_time <= 0:
            raise ValueError("時間パラメータは正の実数である必要があります")
        
        if self.time.save_interval > self.time.max_time:
            raise ValueError("保存間隔は最大時間以下である必要があります")
        
        # 相の設定の検証
        if not self.phases:
            raise ValueError("少なくとも1つの相を設定する必要があります")
        
        # 境界条件の検証
        valid_boundaries = {'periodic', 'neumann', 'dirichlet'}
        for bc in [self.boundary.x, self.boundary.y, self.boundary.z]:
            if bc not in valid_boundaries:
                raise ValueError(f"無効な境界条件: {bc}")
        
        # ソルバーの設定の検証
        if self.solver.poisson_tolerance <= 0:
            raise ValueError("Poissonソルバーの許容誤差は正の値である必要があります")
        
        if self.solver.poisson_max_iterations <= 0:
            raise ValueError("最大反復回数は正の整数である必要があります")
    
    def get_dx(self) -> Tuple[float, float, float]:
        """グリッド間隔を取得"""
        return (
            self.grid.lx / self.grid.nx,
            self.grid.ly / self.grid.ny,
            self.grid.lz / self.grid.nz
        )
    
    def get_shape(self) -> Tuple[int, int, int]:
        """グリッド形状を取得"""
        return (self.grid.nx, self.grid.ny, self.grid.nz)
    
    def to_dict(self) -> Dict:
        """設定を辞書形式に変換"""
        return {
            'phases': [vars(phase) for phase in self.phases],
            'grid': vars(self.grid),
            'time': vars(self.time),
            'solver': vars(self.solver),
            'boundary': vars(self.boundary),
            'initial': vars(self.initial),
            'output_dir': self.output_dir
        }
    
    def save(self, filename: str):
        """設定をファイルに保存"""
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, filename: str) -> 'SimulationConfig':
        """設定をファイルから読み込み"""
        config = cls.__new__(cls)
        
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        
        config.phases = [PhaseConfig(**phase) for phase in data['phases']]
        config.grid = GridConfig(**data['grid'])
        config.time = TimeConfig(**data['time'])
        config.solver = SolverConfig(**data['solver'])
        config.boundary = BoundaryConfig(**data['boundary'])
        config.initial = InitialConditionConfig(**data['initial'])
        config.output_dir = data['output_dir']
        
        # 設定の検証
        config._validate()
        
        return config
