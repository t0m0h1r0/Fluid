import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class FlowVisualizer:
    """Class for visualizing flow fields"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing grid parameters
        """
        self.nx = config['nx']
        self.ny = config['ny']
        self.nz = config['nz']
        self.dx = config['dx']
        self.dy = config['dy']
        self.dz = config['dz']
        
        # Create coordinate grids
        self.x = np.linspace(0, self.nx * self.dx, self.nx)
        self.y = np.linspace(0, self.ny * self.dy, self.ny)
        self.z = np.linspace(0, self.nz * self.dz, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
    
    def plot_slice(self, field: np.ndarray, plane: str = 'xz', index: Optional[int] = None,
                  title: str = '', cmap: str = 'viridis') -> None:
        """
        Plot a 2D slice of a 3D field
        
        Parameters
        ----------
        field : ndarray
            3D field to visualize
        plane : str
            Plane to plot ('xy', 'xz', or 'yz')
        index : int, optional
            Index for the slice. If None, uses middle index
        title : str
            Plot title
        cmap : str
            Colormap name
        """
        if index is None:
            if plane == 'xy':
                index = self.nz // 2
            elif plane == 'xz':
                index = self.ny // 2
            else:  # yz
                index = self.nx // 2
        
        plt.figure(figsize=(10, 8))
        
        if plane == 'xy':
            plt.pcolormesh(self.X[:, :, index], self.Y[:, :, index],
                          field[:, :, index], shading='auto', cmap=cmap)
            plt.xlabel('x')
            plt.ylabel('y')
        elif plane == 'xz':
            plt.pcolormesh(self.X[:, index, :], self.Z[:, index, :],
                          field[:, index, :], shading='auto', cmap=cmap)
            plt.xlabel('x')
            plt.ylabel('z')
        else:  # yz
            plt.pcolormesh(self.Y[index, :, :], self.Z[index, :, :],
                          field[index, :, :], shading='auto', cmap=cmap)
            plt.xlabel('y')
            plt.ylabel('z')
        
        plt.colorbar()
        plt.title(title)
        plt.axis('equal')
        plt.show()
    
    def plot_interface(self, phase: np.ndarray, threshold: float = 0.5,
                      title: str = '') -> None:
        """Plot 3D interface between fluids"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create isosurface
        ax.contour3D(self.X, self.Y, self.Z, phase, levels=[threshold],
                    cmap='viridis')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        plt.show()
    
    def plot_velocity_vectors(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                            plane: str = 'xz', index: Optional[int] = None,
                            scale: float = 1.0) -> None:
        """Plot velocity vectors on a 2D slice"""
        if index is None:
            if plane == 'xy':
                index = self.nz // 2
            elif plane == 'xz':
                index = self.ny // 2
            else:  # yz
                index = self.nx // 2
        
        plt.figure(figsize=(10, 8))
        
        if plane == 'xy':
            plt.quiver(self.X[:, :, index], self.Y[:, :, index],
                      u[:, :, index], v[:, :, index], scale=scale)
            plt.xlabel('x')
            plt.ylabel('y')
        elif plane == 'xz':
            plt.quiver(self.X[:, index, :], self.Z[:, index, :],
                      u[:, index, :], w[:, index, :], scale=scale)
            plt.xlabel('x')
            plt.ylabel('z')
        else:  # yz
            plt.quiver(self.Y[index, :, :], self.Z[index, :, :],
                      v[index, :, :], w[index, :, :], scale=scale)
            plt.xlabel('y')
            plt.ylabel('z')
        
        plt.axis('equal')
        plt.title('Velocity vectors')
        plt.show()
    
    def create_animation(self, fields: Dict[str, np.ndarray], 
                        times: np.ndarray, plane: str = 'xz',
                        index: Optional[int] = None) -> animation.FuncAnimation:
        """Create animation of field evolution"""
        fig, axes = plt.subplots(len(fields), 1, figsize=(10, 4*len(fields)))
        if len(fields) == 1:
            axes = [axes]
        
        if index is None:
            if plane == 'xy':
                index = self.nz // 2
            elif plane == 'xz':
                index = self.ny // 2
            else:  # yz
                index = self.nx // 2
        
        plots = []
        for ax, (name, field) in zip(axes, fields.items()):
            if plane == 'xy':
                plot = ax.pcolormesh(self.X[:, :, index], self.Y[:, :, index],
                                   field[0, :, :, index], shading='auto')
            elif plane == 'xz':
                plot = ax.pcolormesh(self.X[:, index, :], self.Z[:, index, :],
                                   field[0, :, index, :], shading='auto')
            else:  # yz
                plot = ax.pcolormesh(self.Y[index, :, :], self.Z[index, :, :],
                                   field[0, index, :, :], shading='auto')
            
            ax.set_title(name)
            plt.colorbar(plot, ax=ax)
            plots.append(plot)
        
        def update(frame):
            for plot, field in zip(plots, fields.values()):
                if plane == 'xy':
                    plot.set_array(field[frame, :-1, :-1, index].ravel())
                elif plane == 'xz':
                    plot.set_array(field[frame, :-1, index, :-1].ravel())
                else:  # yz
                    plot.set_array(field[frame, index, :-1, :-1].ravel())
            return plots
        
        ani = animation.FuncAnimation(fig, update, frames=len(times),
                                    interval=50, blit=True)
        plt.close()
        return ani