import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

class DataWriter:
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_density_field(self, density: np.ndarray, config: dict, basename: str):
        h5_filename = os.path.join(self.output_dir, f"{basename}.h5")
        self._save_density_h5(density, config, h5_filename)
        
        png_filename = os.path.join(self.output_dir, f"{basename}.png")
        self._save_density_plot(density, config, png_filename)
        
        self._save_density_plot_3d(density, config, png_filename)
    
    def _save_density_h5(self, density: np.ndarray, config: dict, filename: str):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('density', data=density)
            f.attrs['Lx'] = config.Lx
            f.attrs['Ly'] = config.Ly
            f.attrs['Lz'] = config.Lz
            f.create_dataset('x', data=np.linspace(0, config.Lx, config.Nx))
            f.create_dataset('y', data=np.linspace(0, config.Ly, config.Ny))
            f.create_dataset('z', data=np.linspace(0, config.Lz, config.Nz))
    
    def _save_density_plot(self, density: np.ndarray, config: dict, filename: str):
        fig = plt.figure(figsize=(15, 5))
        
        slices = [
            (density[:, :, density.shape[2]//2], 'xy-plane (z=1.0)', [0, config.Lx, 0, config.Ly]),
            (density[:, density.shape[1]//2, :], 'xz-plane (y=0.5)', [0, config.Lx, 0, config.Lz]),
            (density[density.shape[0]//2, :, :], 'yz-plane (x=0.5)', [0, config.Ly, 0, config.Lz])
        ]
        
        for i, (slice_data, title, extent) in enumerate(slices, 1):
            ax = fig.add_subplot(1, 3, i)
            im = ax.imshow(slice_data.T, origin='lower', extent=extent, cmap='viridis')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Density [kg/m³]')
            ax.set_xlabel('x [m]' if 'xz' in title or 'xy' in title else 'y [m]')
            ax.set_ylabel('y [m]' if 'xy' in title else 'z [m]')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def _save_density_plot_3d(self, density: np.ndarray, config: dict, filename: str):
        # より大きなフィギュアサイズを設定
        fig = plt.figure(figsize=(12, 16))
        ax = fig.add_subplot(111, projection='3d')
        
        # グリッドの作成
        x = np.linspace(0, config.Lx, config.Nx)
        y = np.linspace(0, config.Ly, config.Ny)
        z = np.linspace(0, config.Lz, config.Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # ドメイン全体を表示するために、密度の最小値と最大値を取得
        min_density = density.min()
        max_density = density.max()
        
        # 等値面を複数作成
        n_surfaces = 5  # 等値面の数
        densities = np.linspace(min_density, max_density, n_surfaces+2)[1:-1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(densities)))
        
        for density_level, color in zip(densities, colors):
            verts, faces = self._marching_cubes(density, density_level, (x, y, z))
            if len(verts) > 0:
                surf = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                                     triangles=faces, color=color, alpha=0.3)
        
        # 箱の描画
        self._plot_box(ax, config.Lx, config.Ly, config.Lz)
        
        # 軸の設定
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Density Distribution')
        
        # 視点の設定
        ax.view_init(elev=20, azim=45)
        
        # 軸の範囲設定（余白を追加）
        margin = 0.1
        ax.set_xlim([-margin, config.Lx + margin])
        ax.set_ylim([-margin, config.Ly + margin])
        ax.set_zlim([-margin, config.Lz + margin])

        # アスペクト比を実際の寸法に合わせて設定
        # 最大の辺を基準にスケーリング
        max_length = max(config.Lx, config.Ly, config.Lz)
        ax.set_box_aspect([
            config.Lx/max_length,
            config.Ly/max_length,
            config.Lz/max_length
        ])
        
        plt.savefig(filename.replace('.png', '_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _marching_cubes(self, volume: np.ndarray, threshold: float, spacing: tuple) -> tuple:
        spacing_array = (
            (spacing[0][1] - spacing[0][0]),
            (spacing[1][1] - spacing[1][0]),
            (spacing[2][1] - spacing[2][0])
        )
        verts, faces, _, _ = measure.marching_cubes(volume, threshold, spacing=spacing_array)
        return verts, faces
        
    def _plot_box(self, ax, Lx, Ly, Lz):
        """シミュレーションドメインの箱を描画"""
        # より細い線で箱を描画
        linewidth = 0.5
        
        # 底面を描画
        ax.plot([0, Lx], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([Lx, Lx], [0, Ly], [0, 0], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([Lx, 0], [Ly, Ly], [0, 0], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([0, 0], [Ly, 0], [0, 0], 'k-', alpha=0.5, linewidth=linewidth)
        
        # 上面を描画
        ax.plot([0, Lx], [0, 0], [Lz, Lz], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([Lx, Lx], [0, Ly], [Lz, Lz], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([Lx, 0], [Ly, Ly], [Lz, Lz], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([0, 0], [Ly, 0], [Lz, Lz], 'k-', alpha=0.5, linewidth=linewidth)
        
        # 垂直な辺を描画
        ax.plot([0, 0], [0, 0], [0, Lz], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([Lx, Lx], [0, 0], [0, Lz], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([Lx, Lx], [Ly, Ly], [0, Lz], 'k-', alpha=0.5, linewidth=linewidth)
        ax.plot([0, 0], [Ly, Ly], [0, Lz], 'k-', alpha=0.5, linewidth=linewidth)