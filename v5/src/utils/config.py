# config.py
from dataclasses import dataclass
from typing import List, Tuple
import yaml
import numpy as np

@dataclass
class VisualizationConfig:
    """可視化の設定"""
    phase_3d_elev: float = 30    # 相場の3D表示の仰角
    phase_3d_azim: float = 45    # 相場の3D表示の方位角
    velocity_3d_elev: float = 45 # 速度場の3D表示の仰角
    velocity_3d_azim: float = 60 # 速度場の3D表示の方位角

@dataclass
class PhaseConfig:
    """相の設定"""
    name: str         # 相の名前
    density: float    # 密度 [kg/m³]
    viscosity: float  # 粘性係数 [Pa·s]

@dataclass
class SphereConfig:
    """球の設定"""
    center: Tuple[float, float, float]  # 中心座標 [m]
    radius: float                       # 半径 [m]
    phase: str                          # 相の名前

@dataclass
class LayerConfig:
    """レイヤーの設定"""
    phase: str          # 相の名前
    z_range: List[float]  # z方向の範囲 [m]

class SimulationConfig:
    def __init__(self, config_file: str):
        """
        シミュレーション設定の初期化
        
        Args:
            config_file: 設定ファイルのパス
        """
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 物理パラメータの設定
        self.phases = [PhaseConfig(**phase) for phase in config['physical']['phases']]
        self.layers = [LayerConfig(**layer) for layer in config['initial_condition']['layers']]
        self.spheres = [SphereConfig(**sphere) for sphere in config['initial_condition']['spheres']]
        
        # 計算領域の設定
        self.Nx = config['domain']['Nx']  # x方向のグリッド数
        self.Ny = config['domain']['Ny']  # y方向のグリッド数
        self.Nz = config['domain']['Nz']  # z方向のグリッド数
        self.Lx = config['domain']['Lx']  # x方向の長さ [m]
        self.Ly = config['domain']['Ly']  # y方向の長さ [m]
        self.Lz = config['domain']['Lz']  # z方向の長さ [m]
        
        # 物理パラメータ
        self.gravity = config['physical']['gravity']                # 重力加速度 [m/s²]
        self.surface_tension = config['physical']['surface_tension']  # 表面張力係数 [N/m]
        
        # 初期条件
        self.initial_velocity = tuple(config['initial_condition']['initial_velocity'])
        
        # 数値パラメータ
        self.dt = config['numerical']['dt']                # 時間刻み幅 [s]
        self.save_interval = config['numerical']['save_interval']  # 保存間隔 [s]
        self.max_time = config['numerical']['max_time']    # 最大計算時間 [s]
        
        # 可視化の設定
        vis_config = config.get('visualization', {})
        phase_3d = vis_config.get('phase_3d', {})
        velocity_3d = vis_config.get('velocity_3d', {})
        
        self.visualization = VisualizationConfig(
            phase_3d_elev=phase_3d.get('elev', 30),
            phase_3d_azim=phase_3d.get('azim', 45),
            velocity_3d_elev=velocity_3d.get('elev', 45),
            velocity_3d_azim=velocity_3d.get('azim', 60)
        )
    
    def validate(self) -> List[str]:
        """設定の検証
        
        Returns:
            エラーメッセージのリスト
        """
        errors = []
        
        # グリッドサイズの検証
        if not all(x > 0 for x in [self.Nx, self.Ny, self.Nz]):
            errors.append("グリッドサイズは正の整数である必要があります")
        
        # 物理サイズの検証
        if not all(x > 0 for x in [self.Lx, self.Ly, self.Lz]):
            errors.append("領域サイズは正の実数である必要があります")
        
        # 時間パラメータの検証
        if not all(x > 0 for x in [self.dt, self.save_interval, self.max_time]):
            errors.append("時間パラメータは正の実数である必要があります")
        
        # 相の検証
        phase_names = set(phase.name for phase in self.phases)
        for layer in self.layers:
            if layer.phase not in phase_names:
                errors.append(f"レイヤー中の相 {layer.phase} が定義されていません")
        
        for sphere in self.spheres:
            if sphere.phase not in phase_names:
                errors.append(f"球中の相 {sphere.phase} が定義されていません")
        
        # レイヤーの範囲検証
        for layer in self.layers:
            if len(layer.z_range) != 2 or layer.z_range[0] >= layer.z_range[1]:
                errors.append(f"レイヤー {layer.phase} の範囲が無効です")
            if layer.z_range[0] < 0 or layer.z_range[1] > self.Lz:
                errors.append(f"レイヤー {layer.phase} が計算領域外にあります")
        
        # 球の位置検証
        for sphere in self.spheres:
            if not (0 <= sphere.center[0] <= self.Lx and
                   0 <= sphere.center[1] <= self.Ly and
                   0 <= sphere.center[2] <= self.Lz):
                errors.append(f"球 {sphere.phase} の中心が計算領域外にあります")
        
        return errors