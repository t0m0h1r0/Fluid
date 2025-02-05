import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_plots(self, fields: dict, time: float, step: int):
        # 相場の可視化
        self._plot_phase_field(fields['phi'], time, step)
        
        # 速度場の可視化
        self._plot_velocity_field(
            fields['u'], fields['v'], fields['w'],
            time, step
        )
        
        # 圧力場の可視化
        self._plot_pressure_field(fields['p'], time, step)
    
    def _plot_phase_field(self, phi: np.ndarray, time: float, step: int):
        fig = plt.figure(figsize=(15, 5))
        
        # xy, xz, yz平面でのスライス
        slices = [
            (phi[:, :, phi.shape[2]//2], 'xy-plane'),
            (phi[:, phi.shape[1]//2, :], 'xz-plane'),
            (phi[phi.shape[0]//2, :, :], 'yz-plane')
        ]
        
        for i, (slice_data, title) in enumerate(slices, 1):
            ax = fig.add_subplot(1, 3, i)
            im = ax.imshow(slice_data.T, origin='lower', cmap='coolwarm')
            ax.set_title(f'Phase Field ({title})')
            plt.colorbar(im, ax=ax)
        
        fig.suptitle(f't = {time:.3f}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/phase_{step:06d}.png')
        plt.close()
    
    def _plot_velocity_field(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                           time: float, step: int):
        fig = plt.figure(figsize=(15, 5))
        
        # 速度の大きさ
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        
        # 3つの断面での可視化
        slices = [
            (vel_mag[:, :, vel_mag.shape[2]//2], 'xy-plane'),
            (vel_mag[:, vel_mag.shape[1]//2, :], 'xz-plane'),
            (vel_mag[vel_mag.shape[0]//2, :, :], 'yz-plane')
        ]
        
        for i, (slice_data, title) in enumerate(slices, 1):
            ax = fig.add_subplot(1, 3, i)
            im = ax.imshow(slice_data.T, origin='lower', cmap='viridis')
            ax.set_title(f'Velocity Magnitude ({title})')
            plt.colorbar(im, ax=ax)
        
        fig.suptitle(f't = {time:.3f}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/velocity_{step:06d}.png')
        plt.close()
    
    def _plot_pressure_field(self, p: np.ndarray, time: float, step: int):
        fig = plt.figure(figsize=(15, 5))
        
        # 3つの断面での可視化
        slices = [
            (p[:, :, p.shape[2]//2], 'xy-plane'),
            (p[:, p.shape[1]//2, :], 'xz-plane'),
            (p[p.shape[0]//2, :, :], 'yz-plane')
        ]
        
        for i, (slice_data, title) in enumerate(slices, 1):
            ax = fig.add_subplot(1, 3, i)
            im = ax.imshow(slice_data.T, origin='lower', cmap='RdBu')
            ax.set_title(f'Pressure ({title})')
            plt.colorbar(im, ax=ax)
        
        fig.suptitle(f't = {time:.3f}')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pressure_{step:06d}.png')
        plt.close()