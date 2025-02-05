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
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(0, config.Lx, config.Nx)
        y = np.linspace(0, config.Ly, config.Ny)
        z = np.linspace(0, config.Lz, config.Nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 密度の閾値を設定（水と窒素の中間値）
        threshold = (config.phases[0].density + config.phases[1].density) / 2
        
        # 界面の抽出
        verts, faces = self._marching_cubes(density, threshold, (x, y, z))
        
        # 3D表示
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                        triangles=faces, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Density Interface')
        
        plt.savefig(filename.replace('.png', '_3d.png'), dpi=300)
        plt.close()

    def _marching_cubes(self, volume: np.ndarray, threshold: float, spacing: tuple) -> tuple:
            spacing_array = (
                (spacing[0][1] - spacing[0][0]),
                (spacing[1][1] - spacing[1][0]),
                (spacing[2][1] - spacing[2][0])
            )
            verts, faces, _, _ = measure.marching_cubes(volume, threshold, spacing=spacing_array)
            return verts, faces