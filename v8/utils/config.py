from dataclasses import dataclass, field
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
class PhysicalConfig:
    """物理パラメータの設定"""
    phases: List[PhaseConfig]
    gravity: float = 9.81

@dataclass
class DomainConfig:
    """計算領域の設定"""
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
    poisson_solver: str = "multigrid"
    poisson_tolerance: float = 1e-6
    poisson_max_iterations: int = 100
    time_integrator: str = "rk4"
    convection_scheme: str = "weno"
    velocity_tolerance: float = 1e-6

@dataclass
class BoundaryConfig:
    """境界条件の設定"""
    x: str
    y: str
    z: str

@dataclass
class InitialConditionOperation:
    """初期条件の操作"""
    type: str
    phase: str
    center: Optional[List[float]] = None
    radius: Optional[float] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    min_point: Optional[List[float]] = None
    max_point: Optional[List[float]] = None
    height: Optional[float] = None
    direction: Optional[str] = None

@dataclass
class InitialVelocityConfig:
    """初期速度場の設定"""
    type: str
    parameters: Dict[str, float]

@dataclass
class InitialConditionConfig:
    """初期条件の設定"""
    base_phase: str
    operations: List[InitialConditionOperation]
    initial_velocity: InitialVelocityConfig

@dataclass
class OutputConfig:
    """出力設定"""
    directory: str
    visualization_interval: float
    checkpoint_interval: float

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
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 各コンポーネントの設定を解析
        self.physical = self._parse_physical(config.get('physical', {}))
        self.domain = self._parse_domain(config.get('domain', {}))
        self.time = self._parse_time(config.get('time', {}))
        self.solver = self._parse_solver(config.get('solver', {}))
        self.boundary = self._parse_boundary(config.get('boundary', {}))
        self.initial_condition = self._parse_initial_condition(
            config.get('initial_condition', {})
        )
        self.output = self._parse_output(config.get('output', {}))
    
    def _parse_physical(self, config: Dict) -> PhysicalConfig:
        """物理パラメータの設定を解析"""
        phases = [
            PhaseConfig(
                name=phase['name'],
                density=float(phase['density']),
                viscosity=float(phase['viscosity']),
                surface_tension=float(phase.get('surface_tension', 0.0))
            )
            for phase in config.get('phases', [])
        ]
        return PhysicalConfig(
            phases=phases,
            gravity=float(config.get('gravity', 9.81))
        )
    
    def _parse_domain(self, config: Dict) -> DomainConfig:
        """計算領域の設定を解析"""
        return DomainConfig(
            nx=int(config['nx']),
            ny=int(config['ny']),
            nz=int(config['nz']),
            lx=float(config['lx']),
            ly=float(config['ly']),
            lz=float(config['lz'])
        )
    
    def _parse_time(self, config: Dict) -> TimeConfig:
        """時間発展の設定を解析"""
        return TimeConfig(
            dt=float(config['dt']),
            max_time=float(config['max_time']),
            save_interval=float(config['save_interval']),
            cfl=float(config.get('cfl', 0.5))
        )
    
    def _parse_solver(self, config: Dict) -> SolverConfig:
        """ソルバーの設定を解析"""
        return SolverConfig(
            poisson_solver=str(config.get('poisson_solver', 'multigrid')),
            poisson_tolerance=float(config.get('poisson_tolerance', 1e-6)),
            poisson_max_iterations=int(config.get('poisson_max_iterations', 100)),
            time_integrator=str(config.get('time_integrator', 'rk4')),
            convection_scheme=str(config.get('convection_scheme', 'weno')),
            velocity_tolerance=float(config.get('velocity_tolerance', 1e-6))
        )
    
    def _parse_boundary(self, config: Dict) -> BoundaryConfig:
        """境界条件の設定を解析"""
        return BoundaryConfig(
            x=str(config.get('x', 'periodic')),
            y=str(config.get('y', 'periodic')),
            z=str(config.get('z', 'neumann'))
        )
    
    def _parse_initial_condition(self, config: Dict) -> InitialConditionConfig:
        """初期条件の設定を解析"""
        # 初期操作の解析
        operations = [
            InitialConditionOperation(
                type=op['type'],
                phase=op['phase'],
                **{k: v for k, v in op.items() 
                   if k not in ['type', 'phase']}
            )
            for op in config.get('operations', [])
        ]
        
        # 初期速度の設定
        velocity_config = config.get('initial_velocity', {
            'type': 'zero',
            'parameters': {'u': 0.0, 'v': 0.0, 'w': 0.0}
        })
        
        return InitialConditionConfig(
            base_phase=config['base_phase'],
            operations=operations,
            initial_velocity=InitialVelocityConfig(
                type=velocity_config['type'],
                parameters=velocity_config['parameters']
            )
        )
    
    def _parse_output(self, config: Dict) -> OutputConfig:
        """出力設定の解析"""
        return OutputConfig(
            directory=str(config.get('directory', 'output')),
            visualization_interval=float(config.get('visualization_interval', 0.1)),
            checkpoint_interval=float(config.get('checkpoint_interval', 0.5))
        )
    
    def get_dx(self) -> Tuple[float, float, float]:
        """グリッド間隔を取得"""
        return (
            self.domain.lx / self.domain.nx,
            self.domain.ly / self.domain.ny,
            self.domain.lz / self.domain.nz
        )
    
    def get_shape(self) -> Tuple[int, int, int]:
        """グリッド形状を取得"""
        return (self.domain.nx, self.domain.ny, self.domain.nz)
    
    def to_dict(self) -> Dict:
        """設定を辞書形式に変換"""
        return {
            'physical': {
                'phases': [
                    {
                        'name': phase.name,
                        'density': phase.density,
                        'viscosity': phase.viscosity,
                        'surface_tension': phase.surface_tension
                    }
                    for phase in self.physical.phases
                ],
                'gravity': self.physical.gravity
            },
            'domain': {
                'nx': self.domain.nx, 'ny': self.domain.ny, 'nz': self.domain.nz,
                'lx': self.domain.lx, 'ly': self.domain.ly, 'lz': self.domain.lz
            },
            'time': {
                'dt': self.time.dt,
                'max_time': self.time.max_time,
                'save_interval': self.time.save_interval,
                'cfl': self.time.cfl
            },
            'solver': {
                'poisson_solver': self.solver.poisson_solver,
                'poisson_tolerance': self.solver.poisson_tolerance,
                'poisson_max_iterations': self.solver.poisson_max_iterations,
                'time_integrator': self.solver.time_integrator,
                'convection_scheme': self.solver.convection_scheme,
                'velocity_tolerance': self.solver.velocity_tolerance
            },
            'boundary': {
                'x': self.boundary.x,
                'y': self.boundary.y,
                'z': self.boundary.z
            },
            'initial_condition': {
                'base_phase': self.initial_condition.base_phase,
                'operations': [
                    {k: v for k, v in op.__dict__.items() if v is not None}
                    for op in self.initial_condition.operations
                ],
                'initial_velocity': {
                    'type': self.initial_condition.initial_velocity.type,
                    'parameters': self.initial_condition.initial_velocity.parameters
                }
            },
            'output': {
                'directory': self.output.directory,
                'visualization_interval': self.output.visualization_interval,
                'checkpoint_interval': self.output.checkpoint_interval
            }
        }
    
    def save(self, filename: str):
        """設定をファイルに保存"""
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
    
    @classmethod
    def load(cls, filename: str) -> 'SimulationConfig':
        """設定をファイルから読み込み"""
        return cls(filename)