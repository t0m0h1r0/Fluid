import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class Visualizer:
    """シミュレーション結果の可視化"""
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams["font.family"] = 'Noto Sans CJK JP'
        
        self.colormaps = {
            'density': 'viridis',
            'velocity': 'coolwarm',
            'pressure': 'RdBu',
            'phase': 'RdYlBu'
        }

    def create_slice_plot(self, field: ScalarField | VectorField, 
                         timestep: int, **kwargs) -> Path:
        """断面プロット作成"""
        fig = plt.figure(figsize=(15, 5))
        
        for i, (axis, label) in enumerate([('xy', 'z'), ('xz', 'y'), ('yz', 'x')]):
            ax = fig.add_subplot(131 + i)
            slice_data = self._get_center_slice(
                field.data[0] if isinstance(field, VectorField) else field.data,
                axis=i
            )
            
            im = ax.imshow(slice_data.T,
                          origin='lower',
                          aspect='equal',
                          cmap=kwargs.get('cmap', 'viridis'))
            
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{field.metadata.name} ({axis}平面)')
        
        path = self._get_output_path(f"{field.metadata.name}_slice_{timestep:06d}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path

    def create_vector_plot(self, field: VectorField, timestep: int, **kwargs) -> Path:
        """ベクトル場プロット作成"""
        fig = plt.figure(figsize=(15, 5))
        
        # ベクトル場の大きさを計算
        magnitude = np.sqrt(sum(v**2 for v in field.data))
        
        for i, (axis_pair, label) in enumerate([
            ((0,1), 'xy'), ((0,2), 'xz'), ((1,2), 'yz')
        ]):
            ax = fig.add_subplot(131 + i)
            
            # 背景の大きさ
            slice_data = self._get_center_slice(magnitude, axis=i)
            im = ax.imshow(slice_data.T,
                          origin='lower',
                          aspect='equal',
                          cmap=kwargs.get('cmap', 'viridis'))
            
            # ベクトル場
            v1 = self._get_center_slice(field.data[axis_pair[0]], axis=i)
            v2 = self._get_center_slice(field.data[axis_pair[1]], axis=i)
            
            stride = kwargs.get('stride', 4)
            scale = kwargs.get('scale', 30)
            
            X, Y = np.meshgrid(
                np.arange(0, v1.shape[0], stride),
                np.arange(0, v1.shape[1], stride)
            )
            
            ax.quiver(X, Y,
                     v1[::stride, ::stride].T,
                     v2[::stride, ::stride].T,
                     scale=scale, alpha=0.5)
            
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{field.metadata.name} ({label}平面)')
        
        path = self._get_output_path(f"{field.metadata.name}_vector_{timestep:06d}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path

    def create_isosurface_plot(self, field: ScalarField, 
                                level: float, timestep: int, **kwargs) -> Path:
        """等値面プロット作成"""
        from skimage import measure
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        try:
            verts, faces, _, _ = measure.marching_cubes(
                field.data,
                level=level,
                spacing=field.dx
            )
            
            # カラーマップの設定（デフォルトは青-赤のグラデーション）
            cmap = kwargs.get('cmap', 'coolwarm')
            color_mapper = cm.ScalarMappable(cmap=cmap)
            
            # 頂点の高さに基づいて色を設定
            z_normalized = (verts[:, 2] - verts[:, 2].min()) / (verts[:, 2].max() - verts[:, 2].min())
            colors = color_mapper.to_rgba(z_normalized)
            
            # 3Dサーフェスプロット
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces,
                            shade=True,
                            cmap=cmap,
                            alpha=0.7)
            
            # アクシスラベルと凡例
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            
            # カラーバーの追加
            plt.colorbar(color_mapper, ax=ax, label='高さ方向の相分率')
            
            ax.set_title(f'{field.metadata.name} (等値面 {level})')
            
        except ValueError as e:
            ax.text(0.5, 0.5, 0.5, "等値面が見つかりません",
                    horizontalalignment='center',
                    verticalalignment='center')
        
        path = self._get_output_path(f"{field.metadata.name}_3d_iso_{timestep:06d}.png")
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        return path

    def create_3d_density_plot(self, field: ScalarField, timestep: int, **kwargs) -> Path:
        """密度の3D可視化"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # グリッドの作成
        dx, dy, dz = field.dx
        X, Y, Z = np.meshgrid(
            np.arange(0, field.data.shape[0]*dx, dx),
            np.arange(0, field.data.shape[1]*dy, dy),
            np.arange(0, field.data.shape[2]*dz, dz),
            indexing='ij'
        )
        
        # データの正規化
        data_normalized = (field.data - field.data.min()) / (field.data.max() - field.data.min())
        
        # 等値面と密度をボリュームレンダリング風に可視化
        cmap = kwargs.get('cmap', 'viridis')
        scatter = ax.scatter(
            X.ravel(), Y.ravel(), Z.ravel(), 
            c=data_normalized.ravel(), 
            cmap=cmap, 
            alpha=0.1,  # 透明度を低く
            s=10  # 点のサイズ
        )
        
        # カラーバーの追加
        plt.colorbar(scatter, ax=ax, label='相分率（正規化）')
        
        # ラベルとタイトル
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'{field.metadata.name} 3D密度分布')
        
        # 視点の設定
        ax.view_init(
            elev=kwargs.get('elev', 30), 
            azim=kwargs.get('azim', 45)
        )
        
        path = self._get_output_path(f"{field.metadata.name}_3d_density_{timestep:06d}.png")
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
        return path

    def create_animation(self, 
                        frames: List[Dict[str, ScalarField | VectorField]],
                        field_name: str,
                        output_name: str,
                        **kwargs) -> Path:
        """アニメーションの作成"""
        fig = plt.figure(figsize=(15, 5))
        
        field = frames[0][field_name]
        is_vector = isinstance(field, VectorField)
        
        def update(frame_idx):
            plt.clf()
            frame = frames[frame_idx]
            field = frame[field_name]
            
            if is_vector:
                self.create_vector_plot(field, frame_idx, **kwargs)
            else:
                self.create_slice_plot(field, frame_idx, **kwargs)
        
        anim = FuncAnimation(fig, update,
                           frames=len(frames),
                           interval=kwargs.get('interval', 100))
        
        path = self._get_output_path(f"{output_name}.gif")
        anim.save(path, writer='pillow')
        plt.close()
        return path

    def _get_center_slice(self, data: np.ndarray, axis: int) -> np.ndarray:
        """中央断面の取得"""
        slices = [slice(None)] * data.ndim
        slices[axis] = data.shape[axis] // 2
        return data[tuple(slices)]

    def _get_output_path(self, filename: str) -> Path:
        """出力ファイルパスの生成"""
        return self.output_dir / filename