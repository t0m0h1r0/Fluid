# config/manager.py
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PhaseConfig:
    """各相の物理パラメータ"""
    name: str
    density: float
    viscosity: float
    surface_tension: Optional[float] = None

@dataclass
class DomainConfig:
    """計算領域の設定"""
    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float

@dataclass
class NumericalConfig:
    """数値計算パラメータ"""
    dt: float = 0.001
    save_interval: float = 0.1
    max_time: float = 1.0
    max_steps: int = 1000
    cfl_factor: float = 0.5
    pressure_tolerance: float = 1e-6
    velocity_tolerance: float = 1e-6

@dataclass
class BoundaryConfig:
    """境界条件の設定"""
    x: str = 'periodic'
    y: str = 'periodic'
    z: str = 'neumann'

@dataclass
class SimulationConfig:
    """シミュレーション全体の設定"""
    phases: List[PhaseConfig]
    domain: DomainConfig
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    initial_conditions: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SimulationConfig':
        """
        YAMLファイルから設定を読み込む
        
        Args:
            config_path: 設定ファイルのパス
        
        Returns:
            SimulationConfig インスタンス
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 相の設定変換
        phases = [
            PhaseConfig(
                name=phase['name'],
                density=phase['density'],
                viscosity=phase['viscosity'],
                surface_tension=phase.get('surface_tension')
            ) for phase in config_dict.get('phases', [])
        ]
        
        # 計算領域の設定
        domain_config = DomainConfig(
            Nx=config_dict['domain']['Nx'],
            Ny=config_dict['domain']['Ny'],
            Nz=config_dict['domain']['Nz'],
            Lx=config_dict['domain']['Lx'],
            Ly=config_dict['domain']['Ly'],
            Lz=config_dict['domain']['Lz']
        )
        
        # 数値計算パラメータ
        numerical_config = NumericalConfig(
            **config_dict.get('numerical', {})
        )
        
        # 境界条件
        boundary_config = BoundaryConfig(
            **config_dict.get('boundary_conditions', {})
        )
        
        return cls(
            phases=phases,
            domain=domain_config,
            numerical=numerical_config,
            boundary=boundary_config,
            initial_conditions=config_dict.get('initial_conditions', {})
        )
    
    def validate(self) -> List[str]:
        """
        設定の検証
        
        Returns:
            エラーメッセージのリスト
        """
        errors = []
        
        # 相の検証
        if not self.phases:
            errors.append("少なくとも1つの相を定義する必要があります")
        
        # 計算領域の検証
        if not all(x > 0 for x in [self.domain.Nx, self.domain.Ny, self.domain.Nz]):
            errors.append("グリッドサイズは正の整数である必要があります")
        
        if not all(x > 0 for x in [self.domain.Lx, self.domain.Ly, self.domain.Lz]):
            errors.append("領域サイズは正の実数である必要があります")
        
        # 数値パラメータの検証
        if not all(x > 0 for x in [self.numerical.dt, self.numerical.save_interval, self.numerical.max_time]):
            errors.append("時間パラメータは正の実数である必要があります")
        
        return errors

# 使用例
def example_usage():
    try:
        # 設定ファイルからの読み込み
        config = SimulationConfig.from_yaml('config/simulation.yaml')
        
        # 設定の検証
        validation_errors = config.validate()
        if validation_errors:
            print("設定エラー:")
            for error in validation_errors:
                print(f"  - {error}")
            return
        
        # シミュレーションの実行
        # ...
    
    except Exception as e:
        print(f"設定の読み込み中にエラーが発生しました: {e}")

# サンプルのYAMLファイル内容
"""
# config/simulation.yaml
phases:
  - name: "water"
    density: 1000.0
    viscosity: 1.0e-3
    surface_tension: 0.07
  - name: "nitrogen"
    density: 1.225
    viscosity: 1.79e-5
    surface_tension: 0.05

domain:
  Nx: 32
  Ny: 32
  Nz: 64
  Lx: 1.0
  Ly: 1.0
  Lz: 2.0

numerical:
  dt: 0.001
  save_interval: 0.1
  max_time: 1.0
  max_steps: 1000
  cfl_factor: 0.5

boundary_conditions:
  x: periodic
  y: periodic
  z: neumann

initial_conditions:
  layers:
    - phase: "water"
      z_range: [0.0, 1.4]
    - phase: "nitrogen"
      z_range: [1.4, 2.0]
  spheres:
    - center: [0.5, 0.5, 0.4]
      radius: 0.2
      phase: "nitrogen"
"""
