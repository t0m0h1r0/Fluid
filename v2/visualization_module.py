import matplotlib.pyplot as plt
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Optional, List

class SimulationVisualizer:
    """シミュレーション結果の可視化"""
    def __init__(self, output_dir: Path):
        """初期化"""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_density_distribution(
        self, 
        density: np.ndarray, 
        step: int, 
        grid_config: Dict[str, float]
    ):
        """密度分布の2D可視化"""
        plt.figure(figsize=(12, 5))
        
        # XZ断面
        plt.subplot(121)
        ny_mid = density.shape[1] // 2
        plt.contourf(
            np.linspace(0, grid_config['Lz'], density.shape[2]),
            np.linspace(0, grid_config['Lx'], density.shape[0]),
            density[:, ny_mid, :],
            levels=20,
            cmap='viridis'
        )
        plt.colorbar(label='密度 [kg/m³]')
        plt.title(f'密度分布 XZ断面 (ステップ {step})')
        plt.xlabel('Z [m]')
        plt.ylabel('X [m]')
        
        # YZ断面
        plt.subplot(122)
        nx_mid = density.shape[0] // 2
        plt.contourf(
            np.linspace(0, grid_config['Lz'], density.shape[2]),
            np.linspace(0, grid_config['Ly'], density.shape[1]),
            density[nx_mid, :, :].T,
            levels=20,
            cmap='viridis'
        )
        plt.colorbar(label='密度 [kg/m³]')
        plt.title(f'密度分布 YZ断面 (ステップ {step})')
        plt.xlabel('Z [m]')
        plt.ylabel('Y [m]')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'density_step_{step:04d}.png')
        plt.close()

    def plot_velocity_field(
        self, 
        u: np.ndarray, 
        v: np.ndarray, 
        w: np.ndarray, 
        step: int, 
        grid_config: Dict[str, float]
    ):
        """速度場の2D可視化"""
        plt.figure(figsize=(12, 5))
        
        # XZ断面の速度ベクトル
        plt.subplot(121)
        ny_mid = u.shape[1] // 2
        Y, Z = np.meshgrid(
            np.linspace(0, grid_config['Ly'], u.shape[1]),
            np.linspace(0, grid_config['Lz'], u.shape[2])
        )
        plt.quiver(
            Z[::2, ::2], 
            Y[::2, ::2],
            w[ny_mid, ::2, ::2].T, 
            u[ny_mid, ::2, ::2].T,
            scale=50
        )
        plt.title(f'速度場 XZ断面 (ステップ {step})')
        plt.xlabel('Z [m]')
        plt.ylabel('Y [m]')
        
        # YZ断面の速度ベクトル
        plt.subplot(122)
        nx_mid = u.shape[0] // 2
        Y, Z = np.meshgrid(
            np.linspace(0, grid_config['Ly'], u.shape[1]),
            np.linspace(0, grid_config['Lz'], u.shape[2])
        )
        plt.quiver(
            Z[::2, ::2], 
            Y[::2, ::2],
            w[nx_mid, ::2, ::2].T, 
            v[nx_mid, ::2, ::2].T,
            scale=50
        )
        plt.title(f'速度場 YZ断面 (ステップ {step})')
        plt.xlabel('Z [m]')
        plt.ylabel('Y [m]')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'velocity_step_{step:04d}.png')
        plt.close()

    def save_simulation_data(
        self, 
        simulation_history: Dict[str, List[np.ndarray]], 
        filename: Optional[str] = None
    ):
        """シミュレーション結果のHDF5保存"""
        if filename is None:
            filename = f'simulation_data_{np.datetime64("now")}.h5'
        
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            for field_name, field_data in simulation_history.items():
                f.create_dataset(field_name, data=np.array(field_data))
        
        return filepath

    def plot_time_series(
        self, 
        simulation_history: Dict[str, List[np.ndarray]], 
        field_name: str
    ):
        """指定フィールドの時系列プロット"""
        plt.figure(figsize=(10, 5))
        
        field_data = np.array(simulation_history[field_name])
        
        # 各時間ステップの統計量をプロット
        plt.plot(
            np.mean(field_data, axis=(1,2,3)), 
            label=f'平均 {field_name}'
        )
        plt.plot(
            np.max(field_data, axis=(1,2,3)), 
            label=f'最大 {field_name}'
        )
        plt.plot(
            np.min(field_data, axis=(1,2,3)), 
            label=f'最小 {field_name}'
        )
        
        plt.title(f'{field_name}の時間発展')
        plt.xlabel('ステップ')
        plt.ylabel(f'{field_name}値')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{field_name}_time_series.png')
        plt.close()
