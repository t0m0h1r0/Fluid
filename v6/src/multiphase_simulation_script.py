# simulations/two_phase_flow.py
import numpy as np
import matplotlib.pyplot as plt

# コア機能のインポート
from core.interfaces import Field
from core.simulation import SimulationManager

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
    NeumannBoundaryCondition
)

# 可視化ユーティリティ
class TwoPhaseFlowSimulation:
    """
    二相流体シミュレーションの実行と可視化
    """
    def __init__(self, grid_size=(64, 64, 128)):
        """
        シミュレーションの初期化
        
        Args:
            grid_size: グリッドのサイズ (Nx, Ny, Nz)
        """
        # グリッドサイズの設定
        self.Nx, self.Ny, self.Nz = grid_size
        
        # 流体物性の定義
        self.water = FluidComponent(
            name="water", 
            density=1000.0, 
            viscosity=1.0e-3, 
            surface_tension=0.072
        )
        self.air = FluidComponent(
            name="air", 
            density=1.225, 
            viscosity=1.81e-5, 
            surface_tension=0.03
        )
        
        # 多成分流体プロパティの初期化
        self.fluid_properties = MultiComponentFluidProperties([self.water, self.air])
        
        # 境界条件の設定
        self.boundary_conditions = MultiAxisBoundaryCondition([
            PeriodicBoundaryCondition(axis=0),  # x方向
            PeriodicBoundaryCondition(axis=1),  # y方向
            NeumannBoundaryCondition(axis=2)    # z方向
        ])
        
        # 数値スキームの初期化
        self.numerical_scheme = CompactFiniteDifferenceScheme()
        
        # 物理モデルの初期化
        self.navier_stokes_model = MultiphaseNavierStokesModel(self.fluid_properties)
        self.phase_field_model = PhaseFieldModel()
        
        # フィールドの初期化
        self.initialize_fields()
    
    def initialize_fields(self):
        """
        初期状態のフィールド生成
        """
        # 位相場の初期化（水と空気の界面）
        self.phase_field = self._create_initial_phase_field()
        
        # 速度場の初期化
        self.velocity = [
            np.zeros((self.Nx, self.Ny, self.Nz)) 
            for _ in range(3)
        ]
        
        # 圧力場の初期化
        self.pressure = np.zeros((self.Nx, self.Ny, self.Nz))
    
    def _create_initial_phase_field(self) -> np.ndarray:
        """
        初期位相場の生成
        
        水と空気の界面を含む初期状態を作成
        
        Returns:
            初期位相場
        """
        # 3Dグリッドの生成
        x = np.linspace(0, 1, self.Nx)
        y = np.linspace(0, 1, self.Ny)
        z = np.linspace(0, 2, self.Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 平面界面の作成（z = 1の位置に界面）
        phase_field = np.tanh((1 - Z) / 0.02)
        
        return phase_field
    
    def run_simulation(self, 
                      total_time: float = 1.0, 
                      dt: float = 0.001, 
                      output_interval: float = 0.1):
        """
        シミュレーションの実行
        
        Args:
            total_time: 総シミュレーション時間
            dt: 時間刻み
            output_interval: 出力間隔
        """
        # 時間発展のパラメータ
        time = 0.0
        step = 0
        
        # 出力用のリスト
        times = []
        phase_fields = []
        
        # シミュレーションループ
        while time < total_time:
            # 時間発展パラメータ
            params = {
                'phase_field': self.phase_field,
                'velocity': self.velocity
            }
            
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
                
                # 進捗の表示
                print(f"Step {step}, Time {time:.4f}")
                print(f"  Phase field range: [{self.phase_field.min():.4f}, {self.phase_field.max():.4f}]")
            
            # 時間と段階の更新
            time += dt
            step += 1
        
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
                extent=[0, 1, 0, 1]
            )
            axs[i].set_title(f'Phase Field at t = {t:.4f}')
            plt.colorbar(im, ax=axs[i])
        
        plt.tight_layout()
        plt.show()

# メイン実行
def main():
    # シミュレーションの実行
    sim = TwoPhaseFlowSimulation(grid_size=(64, 64, 128))
    times, phase_fields = sim.run_simulation(
        total_time=0.5, 
        dt=0.001, 
        output_interval=0.1
    )
    
    # 結果の可視化
    sim.visualize_results(times, phase_fields)

if __name__ == "__main__":
    main()
