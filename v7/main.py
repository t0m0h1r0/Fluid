import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from config.base import Config
from config.validator import ConfigValidator
from core.simulation import MultiPhaseSimulation
from core.material.properties import MaterialManager, FluidProperties
from core.solver.navier_stokes.solver import NavierStokesSolver
from core.solver.time_integrator.adaptive import AdaptiveTimeIntegrator
from core.solver.time_integrator.runge_kutta import RK4
from core.solver.poisson.multigrid import MultigridSolver
from physics.phase_field import PhaseField, PhaseFieldParameters
from data_io.visualizer import Visualizer
from data_io.logging import setup_logging
from data_io.hdf5_io import HDF5IO
import numpy as np

def initialize_logging(config: Config):
    """ロギング設定の初期化"""
    log_dir = Path(config.logging.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # ログファイル名に現在の日時を追加
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"simulation_{timestamp}.log"
    
    # ロギングの設定
    setup_logging(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        log_file=log_file
    )
    
    # 設定情報をログに記録
    logging.info("シミュレーション設定:")
    logging.info(f"計算領域: {config.domain.size}")
    logging.info(f"グリッド解像度: {config.domain.resolution}")
    logging.info(f"相の数: {len(config.physical.phases)}")

def create_material_manager(config: Config) -> MaterialManager:
    """物性値マネージャーの作成"""
    material_manager = MaterialManager()
    
    # 各相の物性値を追加
    for phase_config in config.physical.phases:
        fluid_props = FluidProperties(
            name=phase_config.name,
            density=phase_config.density,
            viscosity=phase_config.viscosity,
            surface_tension=phase_config.surface_tension_coefficient
        )
        material_manager.add_fluid(phase_config.name, fluid_props)
    
    return material_manager

def setup_phase_field(config: Config, material_manager: MaterialManager) -> PhaseField:
    """Phase-Fieldの設定"""
    # Phase-Fieldパラメータの設定
    phase_params = PhaseFieldParameters(
        epsilon=0.01,  # 界面厚さ
        mobility=1.0,  # 移動度
        surface_tension=config.physical.surface_tension
    )
    
    # フィールドのメタデータを作成
    from core.field.metadata import FieldMetadata
    from core.field.scalar_field import ScalarField
    
    metadata = FieldMetadata(
        name='phase',
        unit='-',
        domain_size=config.domain.size,
        resolution=config.domain.resolution
    )
    phase_field_scalar = ScalarField(metadata)
    
    # 初期条件の設定（層と球）
    phase_field_scalar.data = np.full(config.domain.resolution, -1.0)
    
    for layer in config.initial_condition.layers:
        z_min, z_max = layer.z_range
        domain_size = config.domain.size
        resolution = config.domain.resolution
        
        k_min = int(z_min / domain_size[2] * resolution[2])
        k_max = int(z_max / domain_size[2] * resolution[2])
        
        max_density = max(
            material_manager.fluids[ph].density 
            for ph in material_manager.fluids
        )
        phase_density = material_manager.fluids[layer.phase].density
        
        layer_value = 1.0 if phase_density == max_density else -1.0
        phase_field_scalar.data[:, :, k_min:k_max] = layer_value
    
    for sphere in config.initial_condition.spheres:
        # グリッド座標
        X, Y, Z = [
            np.linspace(0, size, resolution[i]) 
            for i, size in enumerate(config.domain.size)
        ]
        X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

        # 球の条件
        r = np.sqrt(
            (X - sphere.center[0])**2 + 
            (Y - sphere.center[1])**2 + 
            (Z - sphere.center[2])**2
        )
        mask = r <= sphere.radius

        max_density = max(
            material_manager.fluids[ph].density 
            for ph in material_manager.fluids
        )
        phase_density = material_manager.fluids[sphere.phase].density

        sphere_value = 1.0 if phase_density == max_density else -1.0
        phase_field_scalar.data[mask] = sphere_value
    
    return PhaseField(phase_field_scalar, phase_params)

def create_simulation(config: Config) -> MultiPhaseSimulation:
    """シミュレーションの初期化"""
    # 物性値マネージャー
    material_manager = create_material_manager(config)
    
    # Phase-Field
    phase_field = setup_phase_field(config, material_manager)
    
    # ポアソンソルバー
    poisson_solver = MultigridSolver(
        tolerance=config.numerical.pressure_tolerance,
        max_iterations=1000
    )
    
    # Navier-Stokesソルバー
    ns_solver = NavierStokesSolver(poisson_solver)
    
    # 時間積分器
    base_integrator = RK4()
    time_integrator = AdaptiveTimeIntegrator(
        base_integrator,
        atol=config.numerical.velocity_tolerance,
        rtol=config.numerical.velocity_tolerance
    )
    
    return MultiPhaseSimulation(
        material_manager=material_manager,
        phase_field=phase_field,
        ns_solver=ns_solver,
        time_integrator=time_integrator,
        config=config
    )

def run_simulation(config: Config):
    """シミュレーションの実行"""
    # ロギングの初期化
    initialize_logging(config)
    
    try:
        # シミュレーションの準備
        simulation = create_simulation(config)
        
        # 入出力の設定
        output_dir = Path(config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        io = HDF5IO(output_dir)
        visualizer = Visualizer(output_dir)
        
        # シミュレーションループ
        current_time = 0.0
        timestep = 0
        next_save_time = config.numerical.save_interval
        
        while (current_time < config.numerical.max_time and 
               timestep < config.numerical.max_steps):
            
            # 時間発展
            #simulation.step(config.numerical.dt)
            
            current_time += config.numerical.dt
            timestep += 1
            
            # 結果の保存と可視化
            if current_time >= next_save_time or timestep == config.numerical.max_steps:
                # フィールドデータの保存
                field_data = simulation.get_field_data()
                for name, field in field_data.items():
                    io.save_field(field, timestep)
                
                # 可視化
                if config.output.visualization:
                    # 相場の可視化
                    phase_plot = visualizer.create_slice_plot(
                        field_data['phase'], timestep,
                        cmap='RdYlBu',
                        elev=config.visualization.phase_3d.elev,
                        azim=config.visualization.phase_3d.azim
                    )
                    
                    # 相場の等値面プロット
                    phase_isosurface = visualizer.create_isosurface_plot(
                        field_data['phase'], 
                        level=0.0,  # 界面の等値面
                        timestep=timestep,
                        cmap='coolwarm'
                    )
                    
                    # 相場の3D密度分布プロット
                    phase_3d_density = visualizer.create_3d_density_plot(
                        field_data['phase'], 
                        timestep=timestep,
                        cmap='viridis',
                        elev=config.visualization.phase_3d.elev,
                        azim=config.visualization.phase_3d.azim
                    )
                    
                    # 速度場の可視化
                    velocity_plot = visualizer.create_vector_plot(
                        field_data['velocity'], timestep,
                        cmap='coolwarm',
                        scale=50,
                        elev=config.visualization.velocity_3d.elev,
                        azim=config.visualization.velocity_3d.azim
                    )
                    
                    logging.info(f"可視化: {phase_plot}, {phase_isosurface}, {phase_3d_density}, {velocity_plot}")

                next_save_time += config.numerical.save_interval
                logging.info(f"時刻 {current_time:.3f} s の結果を保存しました")
        
        logging.info("シミュレーションが完了しました")
    
    except Exception as e:
        logging.error(f"シミュレーション中にエラーが発生しました: {e}")
        raise

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='多相流体シミュレーション')
    parser.add_argument(
        '--config', 
        type=Path, 
        default='config.yaml',
        help='設定ファイルのパス'
    )
    args = parser.parse_args()
    
    # 設定ファイルの読み込みとバリデーション
    config = Config.from_yaml(args.config)
    
    # 設定のバリデーション
    validation_errors = ConfigValidator.validate(config)
    if validation_errors:
        for error in validation_errors:
            print(f"設定エラー: {error}")
        return
    
    # シミュレーションの実行
    run_simulation(config)

if __name__ == "__main__":
    main()