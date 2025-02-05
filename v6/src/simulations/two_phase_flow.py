# simulations/two_phase_flow.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging

# プロジェクトのルートディレクトリをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 設定のインポート
from config.simulation_config import SimulationConfig

# 物理モデルのインポート
from physics.multiphase_model import (
    MultiphaseNavierStokesModel, 
    PhaseFieldModel
)
from physics.fluid_properties import (
    FluidComponent, 
    MultiComponentFluidProperties
)

# 数値スキームのインポート
from numerics.schemes import CompactFiniteDifferenceScheme
from numerics.boundary import (
    MultiAxisBoundaryCondition, 
    PeriodicBoundaryCondition, 
    NeumannBoundaryCondition,
    DirichletBoundaryCondition
)

class TwoPhaseFlowSimulation:
    """
    YAML設定を利用した二相流体シミュレーション
    """
    def __init__(self, config: SimulationConfig):
        """
        シミュレーションの初期化
        
        Args:
            config: シミュレーション設定
        """
        self.config = config
        
        # ロギングの設定
        self._setup_logging()
        
        # 流体物性の定義
        self.fluid_components = [
            FluidComponent(
                name=fluid.name,
                density=fluid.density,
                viscosity=fluid.viscosity,
                surface_tension=fluid.surface_tension
            ) for fluid in config.fluids
        ]
        
        # 多成分流体プロパティの初期化
        self.fluid_properties = MultiComponentFluidProperties(self.fluid_components)
        
        # 境界条件のマッピング
        boundary_map = {
            'periodic': PeriodicBoundaryCondition,
            'neumann': NeumannBoundaryCondition,
            'dirichlet': DirichletBoundaryCondition
        }
        
        # 境界条件の設定
        self.boundary_conditions = MultiAxisBoundaryCondition([
            boundary_map.get(config.boundary.x, PeriodicBoundaryCondition)(axis=0),
            boundary_map.get(config.boundary.y, PeriodicBoundaryCondition)(axis=1),
            boundary_map.get(config.boundary.z, NeumannBoundaryCondition)(axis=2)
        ])
        
        # 数値スキームの初期化
        self.numerical_scheme = CompactFiniteDifferenceScheme(
            time_integration='explicit'  # YAML設定から取得可能にする
        )
        
        # 物理モデルの初期化
        self.navier_stokes_model = MultiphaseNavierStokesModel(
            self.fluid_properties, 
            gravity=config.physical_model.gravity
        )
        
        self.phase_field_model = PhaseFieldModel(
            interface_width=config.physical_model.interface_width,
            mobility=config.physical_model.mobility
        )
        
        # フィールドの初期化
        self.initialize_fields()
    
    def _setup_logging(self):
        """
        ロギングの設定
        """
        # ロガーの初期化
        logging.basicConfig(
            level=logging.DEBUG if self.config.numerical.debug_mode else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_fields(self):
        """
        初期状態のフィールド生成
        """
        # グリッドサイズの取得
        Nx = self.config.domain.Nx
        Ny = self.config.domain.Ny
        Nz = self.config.domain.Nz
        
        # 座標軸の生成
        x = np.linspace(0, self.config.domain.Lx, Nx)
        y = np.linspace(0, self.config.domain.Ly, Ny)
        z = np.linspace(0, self.config.domain.Lz, Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 初期位相場の生成
        if self.config.initial_conditions.phase_interface == 'planar':
            # 平面界面の作成
            interface_pos = self.config.initial_conditions.interface_position
            interface_width = self.config.initial_conditions.interface_width
            
            self.phase_field = np.tanh(
                (interface_pos - Z) / interface_width
            )
        else:
            # 他の界面形状を追加可能（球、楕円など）
            raise ValueError(f"未サポートの界面形状: {self.config.initial_conditions.phase_interface}")
        
        # 速度場の初期化（静止状態）
        self.velocity = [
            np.zeros((Nx, Ny, Nz)) 
            for _ in range(3)
        ]
        
        # 圧力場の初期化
        self.pressure = np.zeros((Nx, Ny, Nz))
        
        # デバッグ出力
        if self.config.numerical.debug_mode:
            self.logger.debug(f"位相場形状: {self.phase_field.shape}")
            self.logger.debug(f"位相場範囲: [{self.phase_field.min():.4f}, {self.phase_field.max():.4f}]")
    
    def run_simulation(self):
        """
        シミュレーションの実行
        """
        # パラメータの取得
        total_time = self.config.numerical.total_time
        dt = self.config.numerical.dt
        output_interval = self.config.numerical.output_interval
        max_steps = self.config.numerical.max_steps or int(total_time / dt)
        
        # 時間発展のパラメータ
        time = 0.0
        step = 0
        
        # 出力用のリスト
        times = []
        phase_fields = []
        
        # シミュレーションループ
        while time < total_time and step < max_steps:
            # 時間発展パラメータ
            params = {
                'phase_field': self.phase_field,
                'velocity': self.velocity
            }
            
            # デバッグ出力
            if self.config.numerical.verbose:
                self.logger.info(f"Step {step}, Time {time:.4f}")
                self.logger.info(f"  Phase field range: [{self.phase_field.min():.4f}, {self.phase_field.max():.4f}]")
            
            # Navier-Stokesモデルによるフラックス計算
            ns_flux = self.navier_stokes_model.compute_flux(
                self.velocity[0], params
            )
            ns_source = self.navier_stokes_model.compute_source_terms(
                self.velocity[0], params
            )
            
            # Phase-Fieldモデルによるフラックス計算
            pf_flux = self.phase_field_model.compute_flux(
                self.phase_field, params
            )
            pf_source = self.phase_field_model.compute_source_terms(
                self.phase_field, {}
            )
            
            # フィールドの更新
            # Navier-Stokes (速度場)
            for i in range(3):
                self.velocity[i] -= dt * (
                    ns_flux[i] + ns_source[i]
                )
            
            # Phase-Field (相場)
            self.phase_field -= dt * (
                pf_flux + pf_source
            )
            
            # 境界条件の適用
            self.velocity[0] = self.boundary_conditions.apply(self.velocity[0])
            self.phase_field = self.boundary_conditions.apply(self.phase_field)
            
            # 結果の記録
            if step % int(output_interval / dt) == 0:
                times.append(time)
                phase_fields.append(self.phase_field.copy())
            
            # 時間と段階の更新
            time += dt
            step += 1
        
        # シミュレーション終了ログ
        self.logger.info(f"シミュレーション終了: 総ステップ数 {step}, 総時間 {time:.4f} s")
        
        return times, phase_fields
    
    def visualize_results(self, times, phase_fields):
        """
        シミュレーション結果の可視化
        
        Args:
            times: 時間リスト
            phase_fields: 位相場のリスト
        """
        # 可視化の設定
        fig, axs = plt.subplots(
            len(phase_fields), 1, 
            figsize=(10, 3*len(phase_fields))
        )
        
        # 各タイムステップでの位相場をプロット
        for i, (t, phase_field) in enumerate(zip(times, phase_fields)):
            # 中央断面の選択
            slice_idx = phase_field.shape[2] // 2
            slice_data = phase_field[:, :, slice_idx]
            
            # プロット
            im = axs[i].imshow(
                slice_data.T, 
                cmap='coolwarm', 
                origin='lower', 
                extent=[
                    0, self.config.domain.Lx, 
                    0, self.config.domain.Ly
                ]
            )
            axs[i].set_title(f'Phase Field at t = {t:.4f} s')
            plt.colorbar(im, ax=axs[i])
        
        plt.tight_layout()
        plt.show()

def main():
    """
    メイン実行関数
    """
    try:
        # 設定ファイルのパス（プロジェクトルートからの相対パス）
        config_path = os.path.join(project_root, 'config', 'simulation.yaml')
        
        # 設定の読み込み
        config = SimulationConfig.from_yaml(config_path)
        
        # 設定の検証
        validation_errors = config.validate()
        if validation_errors:
            print("設定エラー:")
            for error in validation_errors:
                print(f"  - {error}")
            return
        
        # シミュレーションの実行
        sim = TwoPhaseFlowSimulation(config)
        times, phase_fields = sim.run_simulation()
        
        # 結果の可視化
        sim.visualize_results(times, phase_fields)
    
    except Exception as e:
        import traceback
        print("シミュレーション中にエラーが発生しました:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
