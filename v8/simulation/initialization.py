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
        self.shape = config.get_shape()
        
        # グリッドの生成
        self.x = np.linspace(0, config.domain.lx, config.domain.nx)
        self.y = np.linspace(0, config.domain.ly, config.domain.ny)
        self.z = np.linspace(0, config.domain.lz, config.domain.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
    
    def initialize_field(self) -> LevelSetField:
        """Level Set場の初期化"""
        # フィールドの作成
        phi = LevelSetField(self.shape, self.dx)
        
        # 基本相の設定
        base_phase = self.config.initial_condition.base_phase
        phase_ops = self._parse_operations()
        
        # 初期値を設定（基本相を-1、その他を1とする）
        phi.data = np.ones(self.shape)
        
        # 各操作を順に適用
        for op in phase_ops:
            self._apply_operation(phi, op)
        
        # 界面の平滑化
        phi.reinitialize()
        
        return phi
    
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
    
    def _apply_z_range(self, phi: LevelSetField, op: GeometricOperation):
        """z方向の範囲指定による相の設定"""
        z_min = op.parameters['z_min']
        z_max = op.parameters['z_max']
        
        # Level Set関数の更新
        phi.data = np.where(
            (self.Z >= z_min) & (self.Z < z_max),
            -1.0,  # 指定相の内部
            1.0    # 指定相の外部
        )
    
    def _apply_sphere(self, phi: LevelSetField, op: GeometricOperation):
        """球形の相の設定"""
        center = np.array(op.parameters['center'])
        radius = op.parameters['radius']
        
        # 中心からの距離を計算
        distance = np.sqrt(
            (self.X - center[0])**2 +
            (self.Y - center[1])**2 +
            (self.Z - center[2])**2
        )
        
        # Level Set関数の更新（距離関数として設定）
        new_phi = radius - distance
        
        # 既存の場との結合（min/maxで内部/外部を決定）
        if phi.data[0,0,0] > 0:  # 外部が正の場合
            phi.data = np.minimum(phi.data, new_phi)
        else:  # 内部が正の場合
            phi.data = np.maximum(phi.data, -new_phi)
    
    def _apply_box(self, phi: LevelSetField, op: GeometricOperation):
        """直方体領域の相の設定"""
        min_point = np.array(op.parameters['min_point'])
        max_point = np.array(op.parameters['max_point'])
        
        # 各方向の距離関数
        dx = np.maximum(min_point[0] - self.X, self.X - max_point[0])
        dy = np.maximum(min_point[1] - self.Y, self.Y - max_point[1])
        dz = np.maximum(min_point[2] - self.Z, self.Z - max_point[2])
        
        # 全体の距離関数
        distance = np.maximum(np.maximum(dx, dy), dz)
        
        # Level Set関数の更新
        new_phi = -distance
        if phi.data[0,0,0] > 0:
            phi.data = np.minimum(phi.data, new_phi)
        else:
            phi.data = np.maximum(phi.data, -new_phi)
    
    def _apply_cylinder(self, phi: LevelSetField, op: GeometricOperation):
        """円柱形の相の設定"""
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
        # フィールドの作成
        velocity = VectorField(self.config.get_shape(), min(self.config.get_dx()))
        
        # 初期速度の設定
        initial_velocity = self.config.initial_condition.initial_velocity
        
        if initial_velocity.type == 'zero':
            # 全ての速度成分をゼロに設定
            for component in velocity.components:
                component.data = np.zeros_like(component.data)
        elif initial_velocity.type == 'uniform':
            # 一様な速度場を設定
            for i, component in enumerate(velocity.components):
                param_key = ['u', 'v', 'w'][i]
                component.data = np.full_like(component.data, 
                                             initial_velocity.parameters.get(param_key, 0.0))
        else:
            raise ValueError(f"未知の初期速度タイプ: {initial_velocity.type}")
        
        return velocity


class SimulationInitializer:
    """シミュレーション全体の初期化を担当するクラス"""
    
    def __init__(self, config: SimulationConfig):
        """
        Args:
            config: シミュレーション設定
        """
        self.config = config
        self.phase_initializer = PhaseInitializer(config)
        self.velocity_initializer = VelocityInitializer(config)
    
    def initialize(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """シミュレーションの初期化を実行"""
        # フィールドの初期化
        phi = self.phase_initializer.initialize_field()
        velocity = self.velocity_initializer.initialize_velocity()
        
        # TODO: 圧力場など他のフィールドの初期化
        
        # フィールドの辞書
        fields = {
            'phi': phi,
            'velocity': velocity
        }
        
        # ソルバーの初期化（仮のプレースホルダー）
        solvers = {}
        
        return fields, solvers