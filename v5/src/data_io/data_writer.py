# data_io/data_writer.py
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import matplotlib.colors as colors
from typing import Tuple, List
from datetime import datetime

class DataWriter:
    def __init__(self, output_dir: str = 'output'):
        """データ出力クラスの初期化
        
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, 'data')
        self.plot_dir = os.path.join(output_dir, 'plots')
        
        # 出力ディレクトリの作成
        for directory in [self.data_dir, self.plot_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # カラーマップの設定
        self.density_cmap = plt.cm.viridis
        self.velocity_cmap = plt.cm.coolwarm

    def save_state(self, phi: np.ndarray, velocity: List[np.ndarray], 
                  pressure: np.ndarray, step: int):
        """シミュレーション状態を保存
        
        Args:
            phi: 相場
            velocity: 速度場 [u, v, w]のリスト
            pressure: 圧力場
            step: タイムステップ
        """
        # HDF5形式でデータを保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        h5_filename = os.path.join(self.data_dir, f"state_{step:06d}_{timestamp}.h5")
        self._save_state_h5(phi, velocity, pressure, step, h5_filename)
        
        # 可視化データの保存
        self._save_visualization(phi, velocity, pressure, step)

    def _save_state_h5(self, phi: np.ndarray, velocity: List[np.ndarray],
                      pressure: np.ndarray, step: int, filename: str):
        """状態データをHDF5形式で保存"""
        with h5py.File(filename, 'w') as f:
            # メインデータ
            f.create_dataset('phi', data=phi)
            f.create_dataset('pressure', data=pressure)
            
            # 速度場
            vel_group = f.create_group('velocity')
            for i, v in enumerate(['u', 'v', 'w']):
                vel_group.create_dataset(v, data=velocity[i])
            
            # メタデータ
            f.attrs['step'] = step
            f.attrs['timestamp'] = datetime.now().isoformat()
            f.attrs['shape'] = phi.shape

    def _save_visualization(self, phi: np.ndarray, velocity: List[np.ndarray], 
                          pressure: np.ndarray, step: int):
        """可視化データの保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"visualization_{step:06d}_{timestamp}"
        
        # 相場の可視化
        self._save_field_visualization(phi, 'phase', base_filename)
        
        # 速度場の可視化
        velocity_magnitude = np.sqrt(sum(v**2 for v in velocity))
        self._save_field_visualization(velocity_magnitude, 'velocity', base_filename)
        
        # 圧力場の可視化
        self._save_field_visualization(pressure, 'pressure', base_filename)

    def _save_field_visualization(self, field: np.ndarray, field_type: str, 
                                base_filename: str):
        """フィールドの可視化を保存"""
        # 2D断面プロット
        fig = plt.figure(figsize=(15, 5))
        
        # 適切なカラーマップと正規化の選択
        if field_type == 'phase':
            cmap = self.density_cmap
            norm = colors.Normalize(vmin=field.min(), vmax=field.max())
            title = '相場'
        elif field_type == 'velocity':
            cmap = self.velocity_cmap
            norm = colors.Normalize(vmin=0, vmax=field.max())
            title = '速度場'
        else:  # pressure
            cmap = plt.cm.RdBu
            max_abs = max(abs(field.min()), abs(field.max()))
            norm = colors.Normalize(vmin=-max_abs, vmax=max_abs)
            title = '圧力場'
        
        # 3つの断面でのプロット
        for i, (axis, plane) in enumerate([
            (2, 'xy'), (1, 'xz'), (0, 'yz')
        ]):
            ax = fig.add_subplot(1, 3, i+1)
            slice_data = self._get_central_slice(field, axis)
            
            im = ax.imshow(slice_data.T, cmap=cmap, norm=norm,
                          origin='lower', aspect='equal')
            
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{title} ({plane}平面)')
        
        plt.tight_layout()
        
        # プロットの保存
        plot_filename = os.path.join(
            self.plot_dir, 
            f"{base_filename}_{field_type}.png"
        )
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _get_central_slice(self, array: np.ndarray, axis: int) -> np.ndarray:
        """配列の中央断面を取得"""
        slicing = [slice(None)] * array.ndim
        slicing[axis] = array.shape[axis] // 2
        return array[tuple(slicing)]

    def load_state(self, filename: str) -> Tuple[np.ndarray, List[np.ndarray], 
                                                np.ndarray, int]:
        """状態データの読み込み"""
        with h5py.File(filename, 'r') as f:
            phi = f['phi'][...]
            pressure = f['pressure'][...]
            velocity = [
                f['velocity/u'][...],
                f['velocity/v'][...],
                f['velocity/w'][...]
            ]
            step = f.attrs['step']
            
        return phi, velocity, pressure, step

    def save_metadata(self, metadata: dict, step: int):
        """メタデータの保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.data_dir,
            f"metadata_{step:06d}_{timestamp}.h5"
        )
        
        with h5py.File(filename, 'w') as f:
            for key, value in metadata.items():
                if isinstance(value, (np.ndarray, List)):
                    f.create_dataset(key, data=value)
                else:
                    f.attrs[key] = value

    def _create_directory_if_not_exists(self, directory: str):
        """ディレクトリが存在しない場合は作成"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"ディレクトリを作成しました: {directory}")

    def cleanup_old_files(self, max_files: int = 1000):
        """古いファイルの削除"""
        for directory in [self.data_dir, self.plot_dir]:
            files = sorted(os.listdir(directory))
            if len(files) > max_files:
                for old_file in files[:-max_files]:
                    os.remove(os.path.join(directory, old_file))
                print(f"{len(files) - max_files}個の古いファイルを削除しました")