# data_writer.py
import os
import numpy as np
import h5py # type: ignore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure # type: ignore
import matplotlib.colors as colors
from typing import Tuple, List

class DataWriter:
    def __init__(self, output_dir: str = 'output'):
        """データ出力クラスの初期化
        
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # カラーマップの設定
        self.density_cmap = plt.cm.viridis
        self.velocity_cmap = plt.cm.coolwarm
    
    def save_density_field(self, density: np.ndarray, config: dict, basename: str):
        """密度場のデータを保存
        
        Args:
            density: 密度場
            config: シミュレーション設定
            basename: 出力ファイルのベース名
        """
        # HDF5形式でデータを保存
        h5_filename = os.path.join(self.output_dir, f"{basename}.h5")
        self._save_density_h5(density, config, h5_filename)
        
        # 2D断面図を保存
        png_filename = os.path.join(self.output_dir, f"{basename}_slices.png")
        self._save_density_plot(density, config, png_filename)
        
        # 3D可視化を保存
        png_3d_filename = os.path.join(self.output_dir, f"{basename}_3d.png")
        self._save_density_plot_3d(density, config, png_3d_filename)
    
    def _save_density_h5(self, density: np.ndarray, config: dict, filename: str):
        """密度場をHDF5形式で保存"""
        with h5py.File(filename, 'w') as f:
            # メインデータ
            f.create_dataset('density', data=density)
            
            # メタデータ
            f.attrs['Lx'] = config.Lx
            f.attrs['Ly'] = config.Ly
            f.attrs['Lz'] = config.Lz
            
            # 座標データ
            f.create_dataset('x', data=np.linspace(0, config.Lx, config.Nx))
            f.create_dataset('y', data=np.linspace(0, config.Ly, config.Ny))
            f.create_dataset('z', data=np.linspace(0, config.Lz, config.Nz))
    
    def _save_density_plot(self, density: np.ndarray, config: dict, filename: str):
        """密度場の2D断面図を保存"""
        fig = plt.figure(figsize=(15, 5))
        
        # 正規化されたカラーマップの作成
        vmin, vmax = density.min(), density.max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # 3つの断面でのプロット
        slices = [
            (density[:, :, density.shape[2]//2], 'xy-plane (z=0.5)', [0, config.Lx, 0, config.Ly]),
            (density[:, density.shape[1]//2, :], 'xz-plane (y=0.5)', [0, config.Lx, 0, config.Lz]),
            (density[density.shape[0]//2, :, :], 'yz-plane (x=0.5)', [0, config.Ly, 0, config.Lz])
        ]
        
        for i, (slice_data, title, extent) in enumerate(slices, 1):
            ax = fig.add_subplot(1, 3, i)
            im = ax.imshow(
                slice_data.T,
                origin='lower',
                extent=extent,
                cmap=self.density_cmap,
                norm=norm
            )
            ax.set_title(title)
            ax.set_xlabel('x [m]' if 'xz' in title or 'xy' in title else 'y [m]')
            ax.set_ylabel('y [m]' if 'xy' in title else 'z [m]')
            plt.colorbar(im, ax=ax, label='密度 [kg/m³]')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _save_density_plot_3d(self, density: np.ndarray, config: dict, filename: str):
        """密度場の3D可視化を保存"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # グリッドの作成
        x = np.linspace(0, config.Lx, config.Nx)
        y = np.linspace(0, config.Ly, config.Ny)
        z = np.linspace(0, config.Lz, config.Nz)
        
        # 等値面の作成
        vmin, vmax = density.min(), density.max()
        levels = np.linspace(vmin, vmax, 5)[1:-1]
        
        # 各等値面の描画
        for level in levels:
            verts, faces = self._marching_cubes(density, level, (x, y, z))
            if len(verts) > 0:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
                color = self.density_cmap(norm(level))
                mesh = ax.plot_trisurf(
                    verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces,
                    color=color,
                    alpha=0.3
                )
        
        # 計算領域の箱を描画
        self._draw_domain_box(ax, config.Lx, config.Ly, config.Lz)
        
        # 視点と軸の設定
        ax.view_init(elev=config.visualization.phase_3d_elev,
                    azim=config.visualization.phase_3d_azim)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        
        # アスペクト比の設定
        max_length = max(config.Lx, config.Ly, config.Lz)
        ax.set_box_aspect([
            config.Lx/max_length,
            config.Ly/max_length,
            config.Lz/max_length
        ])
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _marching_cubes(self, volume: np.ndarray, level: float, 
                       spacing: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """マーチングキューブ法による等値面の抽出"""
        spacing_array = (
            spacing[0][1] - spacing[0][0],
            spacing[1][1] - spacing[1][0],
            spacing[2][1] - spacing[2][0]
        )
        verts, faces, _, _ = measure.marching_cubes(
            volume, level,
            spacing=spacing_array,
            allow_degenerate=False
        )
        return verts, faces

    def _draw_domain_box(self, ax: plt.Axes, Lx: float, Ly: float, Lz: float):
        """計算領域の箱を描画"""
        # より細い線で箱を描画
        linewidth = 0.5
        alpha = 0.3
        
        # 底面
        ax.plot([0, Lx], [0, 0], [0, 0], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([Lx, Lx], [0, Ly], [0, 0], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([Lx, 0], [Ly, Ly], [0, 0], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([0, 0], [Ly, 0], [0, 0], 'k-', alpha=alpha, linewidth=linewidth)
        
        # 上面
        ax.plot([0, Lx], [0, 0], [Lz, Lz], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([Lx, Lx], [0, Ly], [Lz, Lz], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([Lx, 0], [Ly, Ly], [Lz, Lz], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([0, 0], [Ly, 0], [Lz, Lz], 'k-', alpha=alpha, linewidth=linewidth)
        
        # 垂直な辺
        ax.plot([0, 0], [0, 0], [0, Lz], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([Lx, Lx], [0, 0], [0, Lz], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([Lx, Lx], [Ly, Ly], [0, Lz], 'k-', alpha=alpha, linewidth=linewidth)
        ax.plot([0, 0], [Ly, Ly], [0, Lz], 'k-', alpha=alpha, linewidth=linewidth)
        
    def save_velocity_field(self, velocity: List[np.ndarray], config: dict, basename: str):
        """速度場のデータを保存
        
        Args:
            velocity: 速度場 [u, v, w]のリスト
            config: シミュレーション設定
            basename: 出力ファイルのベース名
        """
        # HDF5形式でデータを保存
        h5_filename = os.path.join(self.output_dir, f"{basename}.h5")
        with h5py.File(h5_filename, 'w') as f:
            for i, v in enumerate(['u', 'v', 'w']):
                f.create_dataset(f'velocity_{v}', data=velocity[i])
        
        # 速度場の可視化を保存
        png_filename = os.path.join(self.output_dir, f"{basename}_velocity.png")
        self._save_velocity_plot(velocity, config, png_filename)