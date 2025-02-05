import os
import numpy as np
import h5py # type: ignore
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure # type: ignore

class DataWriter:
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_density_field(self, field: np.ndarray, config: dict, basename: str):
        h5_filename = os.path.join(self.output_dir, f"{basename}.h5")
        self._save_field_h5(field, config, h5_filename)
        
        png_filename = os.path.join(self.output_dir, f"{basename}.png")
        self._save_field_plot(field, config, png_filename)
        
        self._save_field_plot_3d(
            field, 
            config, 
            png_filename, 
            elev=config.visualization.phase_3d_elev, 
            azim=config.visualization.phase_3d_azim
        )
    
    def save_velocity_field(self, velocity: list, config: dict, basename: str):
        """速度場の保存"""
        h5_filename = os.path.join(self.output_dir, f"{basename}_velocity.h5")
        
        # HDF5ファイルに速度場を保存
        with h5py.File(h5_filename, 'w') as f:
            f.create_dataset('u', data=velocity[0])
            f.create_dataset('v', data=velocity[1])
            f.create_dataset('w', data=velocity[2])
            f.attrs['Lx'] = config.Lx
            f.attrs['Ly'] = config.Ly
            f.attrs['Lz'] = config.Lz
            f.create_dataset('x', data=np.linspace(0, config.Lx, config.Nx))
            f.create_dataset('y', data=np.linspace(0, config.Ly, config.Ny))
            f.create_dataset('z', data=np.linspace(0, config.Lz, config.Nz))
        
        # 速度の大きさを計算
        vel_mag = np.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
        
        png_filename = os.path.join(self.output_dir, f"{basename}_velocity.png")
        
        # 2Dスライスの可視化
        fig = plt.figure(figsize=(15, 5))
        
        slices = [
            (vel_mag[:, :, vel_mag.shape[2]//2], 'xy-plane (z=1.0)', [0, config.Lx, 0, config.Ly]),
            (vel_mag[:, vel_mag.shape[1]//2, :], 'xz-plane (y=0.5)', [0, config.Lx, 0, config.Lz]),
            (vel_mag[vel_mag.shape[0]//2, :, :], 'yz-plane (x=0.5)', [0, config.Ly, 0, config.Lz])
        ]
        
        for i, (slice_data, title, extent) in enumerate(slices, 1):
            ax = fig.add_subplot(1, 3, i)
            im = ax.imshow(slice_data.T, origin='lower', extent=extent, cmap='viridis')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Velocity Magnitude [m/s]')
            ax.set_xlabel('x [m]' if 'xz' in title or 'xy' in title else 'y [m]')
            ax.set_ylabel('y [m]' if 'xy' in title else 'z [m]')
        
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300)
        plt.close()
        
        # 3D可視化用の速度の大きさ
        self._save_velocity_plot_3d(
            vel_mag, 
            config, 
            png_filename, 
            elev=config.visualization.velocity_3d_elev, 
            azim=config.visualization.velocity_3d_azim
        )
    
    def _save_velocity_plot_3d(self, vel_mag: np.ndarray, config: dict, filename: str, 
                               elev: float = 45, azim: float = 60):
        """
        速度場の3D可視化プロット
        
        Args:
            vel_mag (np.ndarray): 速度の大きさ
            config (dict): シミュレーション設定
            filename (str): 出力ファイル名
            elev (float, optional): 仰角（垂直方向の角度）. Defaults to 45.
            azim (float, optional): 方位角（水平方向の角度）. Defaults to 60.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(0, config.Lx, config.Nx)
        y = np.linspace(0, config.Ly, config.Ny)
        z = np.linspace(0, config.Lz, config.Nz)
        
        # 速度の大きさのしきい値を平均値に設定
        threshold = np.mean(vel_mag)
        
        # 界面の抽出
        verts, faces = self._marching_cubes(vel_mag, threshold, (x, y, z))
        
        # 3D表示
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                        triangles=faces, cmap='viridis', alpha=0.8)
        
        # 視点の設定
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Velocity Magnitude')
        
        plt.savefig(filename.replace('.png', '_3d.png'), dpi=300)
        plt.close()
    
    def _save_field_h5(self, field: np.ndarray, config: dict, filename: str):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('field', data=field)
            f.attrs['Lx'] = config.Lx
            f.attrs['Ly'] = config.Ly
            f.attrs['Lz'] = config.Lz
            f.create_dataset('x', data=np.linspace(0, config.Lx, config.Nx))
            f.create_dataset('y', data=np.linspace(0, config.Ly, config.Ny))
            f.create_dataset('z', data=np.linspace(0, config.Lz, config.Nz))
    
    def _save_field_plot(self, field: np.ndarray, config: dict, filename: str):
        fig = plt.figure(figsize=(15, 5))
        
        slices = [
            (field[:, :, field.shape[2]//2], 'xy-plane (z=1.0)', [0, config.Lx, 0, config.Ly]),
            (field[:, field.shape[1]//2, :], 'xz-plane (y=0.5)', [0, config.Lx, 0, config.Lz]),
            (field[field.shape[0]//2, :, :], 'yz-plane (x=0.5)', [0, config.Ly, 0, config.Lz])
        ]
        
        for i, (slice_data, title, extent) in enumerate(slices, 1):
            ax = fig.add_subplot(1, 3, i)
            im = ax.imshow(slice_data.T, origin='lower', extent=extent, cmap='coolwarm', vmin=0, vmax=1)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Phase Field')
            ax.set_xlabel('x [m]' if 'xz' in title or 'xy' in title else 'y [m]')
            ax.set_ylabel('y [m]' if 'xy' in title else 'z [m]')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def _save_field_plot_3d(self, field: np.ndarray, config: dict, filename: str, 
                            elev: float = 30, azim: float = 45):
        """
        3D可視化プロット
        
        Args:
            field (np.ndarray): 可視化する3Dデータ
            config (dict): シミュレーション設定
            filename (str): 出力ファイル名
            elev (float, optional): 仰角（垂直方向の角度）. Defaults to 30.
            azim (float, optional): 方位角（水平方向の角度）. Defaults to 45.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(0, config.Lx, config.Nx)
        y = np.linspace(0, config.Ly, config.Ny)
        z = np.linspace(0, config.Lz, config.Nz)
        
    def _save_field_plot_3d(self, field: np.ndarray, config: dict, filename: str, 
                            elev: float = 30, azim: float = 45):
        """
        3D可視化プロット
        
        Args:
            field (np.ndarray): 可視化する3Dデータ
            config (dict): シミュレーション設定
            filename (str): 出力ファイル名
            elev (float, optional): 仰角（垂直方向の角度）. Defaults to 30.
            azim (float, optional): 方位角（水平方向の角度）. Defaults to 45.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 各軸の実際の長さ
        x = np.linspace(0, config.Lx, config.Nx)
        y = np.linspace(0, config.Ly, config.Ny)
        z = np.linspace(0, config.Lz, config.Nz)
        
        # しきい値を0.5に設定（相場の中間値）
        threshold = 0.5
        
        # 界面の抽出
        verts, faces = self._marching_cubes(field, threshold, (x, y, z))
        
        # 3D表示
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], 
                        triangles=faces, cmap='coolwarm', alpha=0.8)
        
        # 視点の設定
        ax.view_init(elev=elev, azim=azim)
        
        # 軸の設定
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Phase Field Interface')
        
        # アスペクト比の調整
        ax.set_box_aspect((config.Lx, config.Ly, config.Lz))
        
        plt.tight_layout()
        plt.savefig(filename.replace('.png', '_3d.png'), dpi=300)
        plt.close()

    def _save_velocity_plot_3d(self, vel_mag: np.ndarray, config: dict, filename: str, 
                               elev: float = 45, azim: float = 60):
        """
        速度場の3D可視化プロット
        
        Args:
            vel_mag (np.ndarray): 速度の大きさ
            config (dict): シミュレーション設定
            filename (str): 出力ファイル名
            elev (float, optional): 仰角（垂直方向の角度）. Defaults to 45.
            azim (float, optional): 方位角（水平方向の角度）. Defaults to 60.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(0, config.Lx, config.Nx)
        y = np.linspace(0, config.Ly, config.Ny)
        z = np.linspace(0, config.Lz, config.Nz)
        
        # 速度の大きさの可視化方法を変更
        # カラーマップを使用してデータを表示
        # マーチングキューブスの代わりに等高線プロットを使用
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 3D散布図で速度の大きさを可視化
        scatter = ax.scatter(X, Y, Z, c=vel_mag, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, ax=ax, label='Velocity Magnitude [m/s]')
        
        # 視点の設定
        ax.view_init(elev=elev, azim=azim)
        
        # 軸の設定
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('Velocity Magnitude')
        
        # アスペクト比の調整
        ax.set_box_aspect((config.Lx, config.Ly, config.Lz))
        
        plt.tight_layout()
        plt.savefig(filename.replace('.png', '_3d.png'), dpi=300)
        plt.close()

    def _marching_cubes(self, volume: np.ndarray, threshold: float, spacing: tuple) -> tuple:
            """
            マーチングキューブス法による界面抽出
            
            Args:
                volume (np.ndarray): 3D配列（相場）
                threshold (float): 界面のしきい値
                spacing (tuple): 各次元の間隔
            
            Returns:
                tuple: 頂点と面のデータ
            """
            spacing_array = (
                (spacing[0][1] - spacing[0][0]),
                (spacing[1][1] - spacing[1][0]),
                (spacing[2][1] - spacing[2][0])
            )
            
            # データ範囲内にしきい値があることを確認
            data_min, data_max = np.min(volume), np.max(volume)
            
            # しきい値が範囲外の場合は、範囲内の中央値を使用
            if threshold < data_min or threshold > data_max:
                threshold = (data_min + data_max) / 2
            
            # マーチングキューブスの実行
            verts, faces, _, _ = measure.marching_cubes(volume, threshold, spacing=spacing_array)
            return verts, faces