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
        
        # サブサンプリングの間隔（密度を下げる）
        subsample = 4
        
        # 速度の大きさと方向
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        
        # 3つの断面での可視化
        slices = [
            (u[:, :, u.shape[2]//2], v[:, :, u.shape[2]//2], w[:, :, u.shape[2]//2], 'xy-plane', 2),
            (u[:, u.shape[1]//2, :], v[:, u.shape[1]//2, :], w[:, u.shape[1]//2, :], 'xz-plane', 0),
            (u[u.shape[0]//2, :, :], v[u.shape[0]//2, :, :], w[u.shape[0]//2, :, :], 'yz-plane', 1)
        ]
        
        for i, (u_slice, v_slice, w_slice, title, plane_idx) in enumerate(slices, 1):
            ax = fig.add_subplot(1, 3, i)
            
            # グリッドの作成
            if plane_idx == 2:  # xy plane
                y, x = np.meshgrid(
                    np.arange(0, u_slice.shape[1], subsample),
                    np.arange(0, u_slice.shape[0], subsample),
                    indexing='ij'
                )
            elif plane_idx == 0:  # xz plane
                y, x = np.meshgrid(
                    np.arange(0, u_slice.shape[1], subsample),
                    np.arange(0, u_slice.shape[0], subsample),
                    indexing='ij'
                )
            else:  # yz plane
                y, x = np.meshgrid(
                    np.arange(0, u_slice.shape[1], subsample),
                    np.arange(0, u_slice.shape[0], subsample),
                    indexing='ij'
                )
            
            # 対応するベクトルの抽出
            u_plot = u_slice[::subsample, ::subsample]
            v_plot = v_slice[::subsample, ::subsample]
            
            # 速度ベクトルの色を速さで変更
            vel_slice = np.sqrt(u_plot**2 + v_plot**2)
            
            # ベクトルプロット
            q = ax.quiver(x, y, u_plot, v_plot, vel_slice, 
                          cmap='viridis', scale=10, pivot='mid')
            plt.colorbar(q, ax=ax, label='速度 [m/s]')
            
            ax.set_title(f'Velocity Field ({title})')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
        
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