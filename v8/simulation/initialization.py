import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from core.field import Field, VectorField
from physics.level_set import LevelSetField
from utils.config import SimulationConfig

@dataclass
class GeometricOperation:
    """相の幾何学的設定操作を表現するクラス"""
    type: str
    phase: str
    parameters: Dict[str, Any]

class PhaseInitializer:
    """相の初期化を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        """
        Args:
            config: シミュレーション設定
        """
        self.config = config
        dx_values = config.get_dx()
        self.dx = min(dx_values)  # 等方的なグリッドを仮定
        
        # グリッドの形状を明示的に取得
        self.nx, self.ny, self.nz = config.domain.nx, config.domain.ny, config.domain.nz
        
        # グリッドの生成（順序に注意）
        dx, dy, dz = config.get_dx()
        x = np.linspace(0, config.domain.lx, self.nx, endpoint=False)
        y = np.linspace(0, config.domain.ly, self.ny, endpoint=False)
        z = np.linspace(0, config.domain.lz, self.nz, endpoint=False)
        
        # meshgridを正しい順序で生成（indexing='ij'を使用）
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        
        # デバッグ情報
        print(f"Initialized grid shapes - X: {self.X.shape}, Y: {self.Y.shape}, Z: {self.Z.shape}")
        print(f"Grid ranges - X: [{x.min()}, {x.max()}], Y: [{y.min()}, {y.max()}], Z: [{z.min()}, {z.max()}]")

    def initialize_field(self) -> LevelSetField:
        """Level Set場の初期化"""
        try:
            print(f"Creating field with shape: ({self.nx}, {self.ny}, {self.nz})")
            
            # フィールドの作成
            phi = LevelSetField((self.nx, self.ny, self.nz), self.dx)
            
            # 基本相の設定
            base_phase = self.config.initial_condition.base_phase
            phase_ops = self._parse_operations()
            
            # 初期値を設定（基本相を-1、その他を1とする）
            phi.data = np.ones((self.nx, self.ny, self.nz))
            
            # 各操作を順に適用
            for op in phase_ops:
                print(f"Applying operation: {op.type}")
                self._apply_operation(phi, op)
            
            # 界面の平滑化
            print("Reinitializing field...")
            phi._reinitialize()
            
            print(f"Final field shape: {phi.data.shape}")
            return phi
            
        except Exception as e:
            print(f"Error in initialize_field: {str(e)}")
            print(f"Current shapes - X: {self.X.shape}, Y: {self.Y.shape}, Z: {self.Z.shape}")
            raise
    
    def _parse_operations(self) -> List[GeometricOperation]:
        """設定から操作のリストを生成"""
        operations = []
        
        for op_config in self.config.initial_condition.operations:
            operations.append(
                GeometricOperation(
                    type=op_config.type,
                    phase=op_config.phase,
                    parameters={
                        k: v for k, v in op_config.__dict__.items()
                        if k not in ['type', 'phase'] and v is not None
                    }
                )
            )
        
        return operations
    
    def _apply_operation(self, phi: LevelSetField, op: GeometricOperation):
        """相操作の適用"""
        try:
            if op.type == "z_range":
                self._apply_z_range(phi, op)
            elif op.type == "sphere":
                self._apply_sphere(phi, op)
            elif op.type == "box":
                self._apply_box(phi, op)
            elif op.type == "cylinder":
                self._apply_cylinder(phi, op)
            else:
                raise ValueError(f"未知の操作タイプ: {op.type}")
        except Exception as e:
            print(f"Error in operation {op.type}: {str(e)}")
            print(f"Operation parameters: {op.parameters}")
            raise
    
    def _apply_z_range(self, phi: LevelSetField, op: GeometricOperation):
        """z方向の範囲指定による相の設定"""
        try:
            z_min = op.parameters['z_min']
            z_max = op.parameters['z_max']
            
            # Z配列の形状を確認
            print(f"Z array shape before operation: {self.Z.shape}")
            
            # 条件を計算
            condition = (self.Z >= z_min) & (self.Z < z_max)
            print(f"Condition array shape: {condition.shape}")
            
            # Level Set関数の更新
            phi.data = np.where(condition, -1.0, 1.0)
            print(f"Updated phi shape: {phi.data.shape}")
            
        except Exception as e:
            print(f"Error in z_range operation: {str(e)}")
            print(f"z_min: {z_min}, z_max: {z_max}")
            raise
    
    def _apply_sphere(self, phi: LevelSetField, op: GeometricOperation):
        """球形の相の設定"""
        try:
            center = np.array(op.parameters['center'])
            radius = op.parameters['radius']
            
            # 距離計算
            distance = np.sqrt(
                (self.X - center[0])**2 +
                (self.Y - center[1])**2 +
                (self.Z - center[2])**2
            )
            
            # Level Set関数の更新
            new_phi = radius - distance
            
            # 既存の場との結合
            if phi.data[0,0,0] > 0:  # 外部が正の場合
                phi.data = np.minimum(phi.data, new_phi)
            else:  # 内部が正の場合
                phi.data = np.maximum(phi.data, -new_phi)
                
        except Exception as e:
            print(f"Error in sphere operation: {str(e)}")
            print(f"Center: {center}, Radius: {radius}")
            raise
    
    def _apply_box(self, phi: LevelSetField, op: GeometricOperation):
        """直方体領域の相の設定"""
        try:
            min_point = np.array(op.parameters['min_point'])
            max_point = np.array(op.parameters['max_point'])
            
            # 各方向の距離関数を計算
            dx = np.maximum(min_point[0] - self.X, self.X - max_point[0])
            dy = np.maximum(min_point[1] - self.Y, self.Y - max_point[1])
            dz = np.maximum(min_point[2] - self.Z, self.Z - max_point[2])
            
            # 全体の距離関数を計算
            distance = np.maximum(np.maximum(dx, dy), dz)
            
            # Level Set関数の更新
            new_phi = -distance
            if phi.data[0,0,0] > 0:
                phi.data = np.minimum(phi.data, new_phi)
            else:
                phi.data = np.maximum(phi.data, -new_phi)
                
        except Exception as e:
            print(f"Error in box operation: {str(e)}")
            print(f"Min point: {min_point}, Max point: {max_point}")
            raise
    
    def _apply_cylinder(self, phi: LevelSetField, op: GeometricOperation):
        """円柱形の相の設定"""
        try:
            center = np.array(op.parameters['center'])
            radius = op.parameters['radius']
            height = op.parameters['height']
            direction = op.parameters.get('direction', 'z')
            
            # 方向に応じた距離計算
            if direction == 'z':
                r_distance = np.sqrt(
                    (self.X - center[0])**2 +
                    (self.Y - center[1])**2
                )
                h_distance = np.maximum(
                    center[2] - height/2 - self.Z,
                    self.Z - (center[2] + height/2)
                )
            elif direction == 'x':
                r_distance = np.sqrt(
                    (self.Y - center[1])**2 +
                    (self.Z - center[2])**2
                )
                h_distance = np.maximum(
                    center[0] - height/2 - self.X,
                    self.X - (center[0] + height/2)
                )
            else:  # direction == 'y'
                r_distance = np.sqrt(
                    (self.X - center[0])**2 +
                    (self.Z - center[2])**2
                )
                h_distance = np.maximum(
                    center[1] - height/2 - self.Y,
                    self.Y - (center[1] + height/2)
                )
            
            # 半径方向と高さ方向の距離関数を組み合わせ
            dr = radius - r_distance
            dh = -h_distance
            distance = np.minimum(dr, dh)
            
            # Level Set関数の更新
            new_phi = distance
            if phi.data[0,0,0] > 0:
                phi.data = np.minimum(phi.data, new_phi)
            else:
                phi.data = np.maximum(phi.data, -new_phi)
                
        except Exception as e:
            print(f"Error in cylinder operation: {str(e)}")
            print(f"Center: {center}, Radius: {radius}, Height: {height}")
            raise

class VelocityInitializer:
    """速度場の初期化を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        """
        Args:
            config: シミュレーション設定
        """
        self.config = config
    
    def initialize_velocity(self) -> VectorField:
        """速度場の初期化"""
        try:
            # グリッドの形状を取得
            nx, ny, nz = self.config.domain.nx, self.config.domain.ny, self.config.domain.nz
            print(f"Initializing velocity field with shape: ({nx}, {ny}, {nz})")
            
            # フィールドの作成
            velocity = VectorField((nx, ny, nz), min(self.config.get_dx()))
            
            # 初期速度の設定
            initial_velocity = self.config.initial_condition.initial_velocity
            print(f"Setting initial velocity type: {initial_velocity.type}")
            
            if initial_velocity.type == 'zero':
                # 全ての速度成分をゼロに設定
                for i, component in enumerate(velocity.components):
                    component.data = np.zeros((nx, ny, nz))
                    print(f"Component {i} initialized with zeros: shape {component.data.shape}")
            
            elif initial_velocity.type == 'uniform':
                # 一様な速度場を設定
                for i, component in enumerate(velocity.components):
                    param_key = ['u', 'v', 'w'][i]
                    value = initial_velocity.parameters.get(param_key, 0.0)
                    component.data = np.full((nx, ny, nz), value)
                    print(f"Component {i} initialized with uniform value {value}: shape {component.data.shape}")
            
            else:
                raise ValueError(f"未知の初期速度タイプ: {initial_velocity.type}")
            
            print("Velocity field initialization completed")
            return velocity
            
        except Exception as e:
            print(f"Error in initialize_velocity: {str(e)}")
            print(f"Config details: {self.config.initial_condition.initial_velocity}")
            raise

class SimulationInitializer:
    """シミュレーション全体の初期化を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        """
        Args:
            config: シミュレーション設定
        """
        self.config = config
        print("Creating phase initializer...")
        self.phase_initializer = PhaseInitializer(config)
        print("Creating velocity initializer...")
        self.velocity_initializer = VelocityInitializer(config)
    
    def initialize(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """シミュレーションの初期化を実行"""
        try:
            print("Starting simulation initialization...")
            
            # フィールドの初期化
            print("Initializing phase field...")
            phi = self.phase_initializer.initialize_field()
            
            print("Initializing velocity field...")
            velocity = self.velocity_initializer.initialize_velocity()
            
            # フィールドの辞書作成
            fields = {
                'phi': phi,
                'velocity': velocity
            }
            print("Fields initialized successfully")
            
            # ソルバーの初期化（仮のプレースホルダー）
            solvers = {}
            print("Initialization completed")
            
            return fields, solvers
            
        except Exception as e:
            print(f"Error during simulation initialization: {str(e)}")
            print("Stack trace:")
            import traceback
            traceback.print_exc()
            raise