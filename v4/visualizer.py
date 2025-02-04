import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self, config):
        self.config = config
        
    def plot_phase_field(self, phi, time):
        """相場の可視化"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x, y, z = np.meshgrid(
            self.config.x,
            self.config.y,
            self.config.z,
            indexing='ij'
        )
        
        # 界面の可視化
        phi_surf = 0.5
        ax.contour(x[:,:,0], y[:,:,0], phi[:,:,0], levels=[phi_surf], colors='b')
        ax.contour(x[:,0,:], z[:,0,:], phi[:,0,:], levels=[phi_surf], colors='r')
        ax.contour(y[0,:,:], z[0,:,:], phi[0,:,:], levels=[phi_surf], colors='g')
        
        ax.set_title(f'Phase Field at t = {time:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        plt.tight_layout()
        return fig
    
    def plot_velocity(self, u, v, w, time):
        """速度場の可視化"""
        fig = plt.figure(figsize=(15, 5))
        
        # 速度の大きさ
        vel_mag = np.sqrt(u**2 + v**2 + w**2)
        
        # xy平面でのスライス
        ax1 = fig.add_subplot(131)
        im1 = ax1.contourf(
            self.config.x,
            self.config.y,
            vel_mag[:,:,self.config.Nz//2].T,
            levels=20
        )
        ax1.set_title('Velocity Magnitude (xy-plane)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)
        
        # xz平面でのスライス
        ax2 = fig.add_subplot(132)
        im2 = ax2.contourf(
            self.config.x,
            self.config.z,
            vel_mag[:,self.config.Ny//2,:].T,
            levels=20
        )
        ax2.set_title('Velocity Magnitude (xz-plane)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        plt.colorbar(im2, ax=ax2)
        
        # yz平面でのスライス
        ax3 = fig.add_subplot(133)
        im3 = ax3.contourf(
            self.config.y,
            self.config.z,
            vel_mag[self.config.Nx//2,:,:].T,
            levels=20
        )
        ax3.set_title('Velocity Magnitude (yz-plane)')
        ax3.set_xlabel('y')
        ax3.set_ylabel('z')
        plt.colorbar(im3, ax=ax3)
        
        fig.suptitle(f'Velocity Field at t = {time:.3f}')
        plt.tight_layout()
        return fig
    
    def plot_velocity_vectors(self, u, v, w, time):
        """速度ベクトルの可視化"""
        fig = plt.figure(figsize=(15, 5))
        
        # xy平面でのベクトル場
        ax1 = fig.add_subplot(131)
        x, y = np.meshgrid(self.config.x, self.config.y)
        ax1.quiver(
            x, y,
            u[:,:,self.config.Nz//2].T,
            v[:,:,self.config.Nz//2].T
        )
        ax1.set_title('Velocity Vectors (xy-plane)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        
        # xz平面でのベクトル場
        ax2 = fig.add_subplot(132)
        x, z = np.meshgrid(self.config.x, self.config.z)
        ax2.quiver(
            x, z,
            u[:,self.config.Ny//2,:].T,
            w[:,self.config.Ny//2,:].T
        )
        ax2.set_title('Velocity Vectors (xz-plane)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('z')
        
        # yz平面でのベクトル場
        ax3 = fig.add_subplot(133)
        y, z = np.meshgrid(self.config.y, self.config.z)
        ax3.quiver(
            y, z,
            v[self.config.Nx//2,:,:].T,
            w[self.config.Nx//2,:,:].T
        )
        ax3.set_title('Velocity Vectors (yz-plane)')
        ax3.set_xlabel('y')
        ax3.set_ylabel('z')
        
        fig.suptitle(f'Velocity Vectors at t = {time:.3f}')
        plt.tight_layout()
        return fig
