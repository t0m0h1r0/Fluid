# config/simulation_config.py
import os
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class FluidComponentConfig:
    """
    単一流体成分の設定
    """
    name: str
    density: float
    viscosity: float
    surface_tension: Optional[float] = None

@dataclass
class DomainConfig:
    """
    計算領域の設定
    """
    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float

@dataclass
class NumericalConfig:
    """
    数値計算パラメータ
    """
    dt: float = 0.001
    total_time: float = 1.0
    output_interval: float = 0.1
    max_steps: Optional[int] = None
    cfl_factor: float = 0.5
    
    # デバッグ設定
    debug_mode: bool = False
    verbose: bool = False

@dataclass
class BoundaryConfig:
    """
    境界条件の設定
    """
    x: str = 'periodic'
    y: str = 'periodic'
    z: str = 'neumann'

@dataclass
class InitialConditionConfig:
    """
    初期状態の設定
    """
    phase_interface: str = 'planar'  # インターフェースの形状
    interface_position: float = 1.0  # インターフェースの位置
    interface_width: float = 0.02  # インターフェースの厚さ

@dataclass
class PhysicalModelConfig:
    """
    物理モデルのパラメータ
    """
    gravity: float = 9.81
    surface_tension: float = 0.07
    mobility: float = 1.0
    interface_width: float = 0.01

@dataclass
class SimulationConfig:
    """
    シミュレーション全体の設定
    """
    fluids: List[FluidComponentConfig]
    domain: DomainConfig
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    initial_conditions: InitialConditionConfig = field(default_factory=InitialConditionConfig)
    physical_model: PhysicalModelConfig = field(default_factory=PhysicalModelConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SimulationConfig':
        """
        YAMLファイルから設定を読み込む
        
        Args:
            config_path: 設定ファイルのパス
        
        Returns:
            SimulationConfig インスタンス
        """
        # ファイルの存在確認
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
        
        # YAMLファイルの読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 流体成分の変換
        fluids = [
            FluidComponentConfig(
                name=fluid['name'],
                density=fluid['density'],
                viscosity=fluid['viscosity'],
                surface_tension=fluid.get('surface_tension')
            ) for fluid in config_dict.get('fluids', [])
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
        
        # 初期条件
        initial_conditions_config = InitialConditionConfig(
            **config_dict.get('initial_conditions', {})
        )
        
        # 物理モデルパラメータ
        physical_model_config = PhysicalModelConfig(
            **config_dict.get('physical_model', {})
        )
        
        return cls(
            fluids=fluids,
            domain=domain_config,
            numerical=numerical_config,
            boundary=boundary_config,
            initial_conditions=initial_conditions_config,
            physical_model=physical_model_config
        )
    
    def validate(self) -> List[str]:
        """
        設定の検証
        
        Returns:
            エラーメッセージのリスト
        """
        errors = []
        
        # 流体成分の検証
        if not self.fluids:
            errors.append("少なくとも1つの流体成分を定義する必要があります")
        
        # 計算領域の検証
        if not all(x > 0 for x in [self.domain.Nx, self.domain.Ny, self.domain.Nz]):
            errors.append("グリッドサイズは正の整数である必要があります")
        
        if not all(x > 0 for x in [self.domain.Lx, self.domain.Ly, self.domain.Lz]):
            errors.append("領域サイズは正の実数である必要があります")
        
        # 数値パラメータの検証
        if not all(x > 0 for x in [self.numerical.dt, self.numerical.total_time]):
            errors.append("時間パラメータは正の実数である必要があります")
        
        return errors

# デモンストレーション関数
def demonstrate_config():
    """
    設定の読み込みとデモンストレーション
    """
    try:
        # デフォルトの設定ファイルパス
        config_path = 'config/simulation.yaml'
        
        # 設定の読み込み
        config = SimulationConfig.from_yaml(config_path)
        
        # 設定の検証
        validation_errors = config.validate()
        if validation_errors:
            print("設定エラー:")
            for error in validation_errors:
                print(f"  - {error}")
            return
        
        # 設定の表示
        print("設定の詳細:")
        print("\n流体成分:")
        for fluid in config.fluids:
            print(f"  {fluid.name}:")
            print(f"    密度: {fluid.density} kg/m³")
            print(f"    粘性係数: {fluid.viscosity} Pa·s")
        
        print("\n計算領域:")
        print(f"  グリッドサイズ: {config.domain.Nx}x{config.domain.Ny}x{config.domain.Nz}")
        print(f"  領域サイズ: {config.domain.Lx}x{config.domain.Ly}x{config.domain.Lz} m")
        
        print("\n数値パラメータ:")
        print(f"  時間刻み: {config.numerical.dt} s")
        print(f"  総シミュレーション時間: {config.numerical.total_time} s")
        print(f"  出力間隔: {config.numerical.output_interval} s")
        
        print("\n境界条件:")
        print(f"  x方向: {config.boundary.x}")
        print(f"  y方向: {config.boundary.y}")
        print(f"  z方向: {config.boundary.z}")
        
        print("\n初期条件:")
        print(f"  界面形状: {config.initial_conditions.phase_interface}")
        print(f"  界面位置: {config.initial_conditions.interface_position}")
        print(f"  界面厚さ: {config.initial_conditions.interface_width}")
        
        print("\n物理モデルパラメータ:")
        print(f"  重力: {config.physical_model.gravity} m/s²")
        print(f"  表面張力係数: {config.physical_model.surface_tension} N/m")
        
    except Exception as e:
        print(f"設定の読み込み中にエラーが発生しました: {e}")

# メイン実行
if __name__ == "__main__":
    demonstrate_config()
