import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

class Visualizer:
    """シミュレーション結果の可視化クラス"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Args:
            output_dir: 出力ディレクトリのパス
        """
        self.output_dir = Path(output_dir)
        self.snapshot_dir = self.output_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # 描画設定
        plt.style.use('default')
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        
        # カラーマップの設定
        self.colormaps = {
            'phase': 'RdBu',
            'velocity': 'viridis',
            'pressure': 'coolwarm',
            'vorticity': 'RdYlBu'
        }
    
    def create_visualization(self, fields: Dict[str, np.ndarray], time: float,
                           step: int, save: bool = True) -> None:
        """可視化の作成と保存
        
        Args:
            fields: 可視化するフィールド
            time: 現在の時刻
            step: 現在のステップ数
            save: 保存するかどうか
        """
        # マルチプロット図の作成
        fig = plt.figure(figsize=(15, 10))
        
        # 2D断面プロット
        self._plot_2d_slices(fig, fields, 231)
        
        # 3D等値面プロット
        self._plot_3d_isosurface(fig, fields, 232)
        
        # 速度場プロット
        self._plot_velocity_field(fig, fields, 233)
        
        # 渦度プロット
        self._plot_vorticity(fig, fields, 234)
        
        # 統計情報プロット
        self._plot_statistics(fig, fields, 235)
        
        # エネルギースペクトルプロット
        self._plot_energy_spectrum(fig, fields, 236)
        
        # タイトルの設定
        fig.suptitle(f"シミュレーション時間: {time:.3f}s\n"
                    f"タイムステップ: {step}")
        
        plt.tight_layout()
        
        if save:
            # 画像として保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.snapshot_dir / f"vis_{step:06d}_{timestamp}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
    
    def _plot_2d_slices(self, fig: plt.Figure, fields: Dict[str, np.ndarray],
                        position: int) -> None:
        """2D断面プロットの作成"""
        ax = fig.add_subplot(position)
        
        # 相場の断面図
        if 'phi' in fields:
            phi = fields['phi']
            slice_z = phi.shape[2] // 2
            phi_slice = phi[:, :, slice_z]
            
            im = ax.imshow(phi_slice.T, cmap=self.colormaps['phase'],
                          origin='lower', aspect='equal')
            plt.colorbar(im, ax=ax, label='相場')
            
            # 速度ベクトルの重ね描き
            if 'velocity' in fields:
                vx, vy = fields['velocity'][0], fields['velocity'][1]
                vx_slice = vx[:, :, slice_z]
                vy_slice = vy[:, :, slice_z]
                
                # ベクトル場を間引いて表示
                stride = 4
                x, y = np.meshgrid(
                    np.arange(0, vx_slice.shape[0], stride),
                    np.arange(0, vx_slice.shape[1], stride)
                )
                ax.quiver(x, y,
                         vx_slice[::stride, ::stride].T,
                         vy_slice[::stride, ::stride].T,
                         scale=50, alpha=0.5)
        
        ax.set_title("XY平面断面図")
    
    def _plot_3d_isosurface(self, fig: plt.Figure, fields: Dict[str, np.ndarray],
                           position: int) -> None:
        """3D等値面プロットの作成"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            from skimage import measure
            
            ax = fig.add_subplot(position, projection='3d')
            
            if 'phi' in fields:
                phi = fields['phi']
                
                # 等値面の計算
                verts, faces, _, _ = measure.marching_cubes(
                    phi, level=0.0,
                    spacing=(1.0, 1.0, 1.0)
                )
                
                # 等値面の描画
                ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                              triangles=faces,
                              cmap=self.colormaps['phase'],
                              alpha=0.8)
                
                # ビューの設定
                ax.view_init(elev=30, azim=45)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title("界面の3D表示")
        
        except ImportError:
            ax = fig.add_subplot(position)
            ax.text(0.5, 0.5, "3D可視化には\nskimageが必要です",
                   ha='center', va='center')
    
    def _plot_velocity_field(self, fig: plt.Figure, fields: Dict[str, np.ndarray],
                           position: int) -> None:
        """速度場のプロット"""
        ax = fig.add_subplot(position)
        
        if 'velocity' in fields:
            # 速度の大きさを計算
            velocity = fields['velocity']
            velocity_mag = np.sqrt(sum(v**2 for v in velocity))
            
            # 中央断面での速度場
            slice_z = velocity_mag.shape[2] // 2
            vel_slice = velocity_mag[:, :, slice_z]
            
            im = ax.imshow(vel_slice.T, cmap=self.colormaps['velocity'],
                          origin='lower', aspect='equal')
            plt.colorbar(im, ax=ax, label='速度の大きさ [m/s]')
        
        ax.set_title("速度場")
    
    def _plot_vorticity(self, fig: plt.Figure, fields: Dict[str, np.ndarray],
                       position: int) -> None:
        """渦度のプロット"""
        ax = fig.add_subplot(position)
        
        if 'velocity' in fields:
            velocity = fields['velocity']
            
            # 渦度の計算
            dudy, dudz = np.gradient(velocity[0], axis=(1, 2))
            dvdx, dvdz = np.gradient(velocity[1], axis=(0, 2))
            dwdx, dwdy = np.gradient(velocity[2], axis=(0, 1))
            
            vorticity_x = dwdy - dvdz
            vorticity_y = dudz - dwdx
            vorticity_z = dvdx - dudy
            
            # 渦度の大きさ
            vorticity_mag = np.sqrt(vorticity_x**2 + vorticity_y**2 + vorticity_z**2)
            
            # 中央断面での渦度
            slice_z = vorticity_mag.shape[2] // 2
            vort_slice = vorticity_mag[:, :, slice_z]
            
            im = ax.imshow(vort_slice.T, cmap=self.colormaps['vorticity'],
                          origin='lower', aspect='equal')
            plt.colorbar(im, ax=ax, label='渦度の大きさ [1/s]')
        
        ax.set_title("渦度")
    
    def _plot_statistics(self, fig: plt.Figure, fields: Dict[str, np.ndarray],
                        position: int) -> None:
        """統計情報のプロット"""
        ax = fig.add_subplot(position)
        
        if 'stats' in fields:
            stats = fields['stats']
            
            # 統計情報のテキスト表示
            stat_text = []
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    stat_text.append(f"{key}: {value:.3g}")
            
            ax.text(0.5, 0.5, '\n'.join(stat_text),
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title("統計情報")
        ax.axis('off')
    
    def _plot_energy_spectrum(self, fig: plt.Figure, fields: Dict[str, np.ndarray],
                            position: int) -> None:
        """エネルギースペクトルのプロット"""
        ax = fig.add_subplot(position)
        
        if 'velocity' in fields:
            velocity = fields['velocity']
            
            # 3D FFTの計算
            u_hat = np.fft.fftn(velocity[0])
            v_hat = np.fft.fftn(velocity[1])
            w_hat = np.fft.fftn(velocity[2])
            
            # パワースペクトル密度
            E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
            
            # 波数の計算
            kx = np.fft.fftfreq(E_hat.shape[0])
            ky = np.fft.fftfreq(E_hat.shape[1])
            kz = np.fft.fftfreq(E_hat.shape[2])
            
            Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
            K = np.sqrt(Kx**2 + Ky**2 + Kz**2)
            
            # シェル平均
            k_bins = np.linspace(0, np.max(K), 50)
            E = np.zeros_like(k_bins[:-1])
            
            for i in range(len(k_bins) - 1):
                mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
                if mask.any():
                    E[i] = np.mean(E_hat[mask])
            
            # プロット
            ax.loglog(k_bins[:-1], E, 'k-', label='E(k)')
            ax.loglog(k_bins[:-1], k_bins[:-1]**(-5/3), 'r--', label='k^(-5/3)')
            ax.set_xlabel('波数 k')
            ax.set_ylabel('E(k)')
            ax.legend()
            ax.grid(True)
        
        ax.set_title("エネルギースペクトル")
