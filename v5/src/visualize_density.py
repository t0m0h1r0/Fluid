import numpy as np
import h5py
import matplotlib.pyplot as plt
import argparse

def visualize_density(filename: str):
    with h5py.File(filename, 'r') as f:
        density = f['density'][:]
        Lx = f.attrs['Lx']
        Ly = f.attrs['Ly']
        Lz = f.attrs['Lz']
        x = f['x'][:]
        y = f['y'][:]
        z = f['z'][:]

    fig = plt.figure(figsize=(15, 5))
    slices = [
        (density[:, :, density.shape[2]//2], 'xy-plane (z=1.0)', [0, Lx, 0, Ly]),
        (density[:, density.shape[1]//2, :], 'xz-plane (y=0.5)', [0, Lx, 0, Lz]),
        (density[density.shape[0]//2, :, :], 'yz-plane (x=0.5)', [0, Ly, 0, Lz])
    ]

    for i, (slice_data, title, extent) in enumerate(slices, 1):
        ax = fig.add_subplot(1, 3, i)
        im = ax.imshow(slice_data.T, origin='lower', extent=extent, cmap='viridis')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Density [kg/m³]')
        ax.set_xlabel('x [m]' if 'xz' in title or 'xy' in title else 'y [m]')
        ax.set_ylabel('y [m]' if 'xy' in title else 'z [m]')

    plt.tight_layout()
    plt.savefig('density_visualization.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the density field HDF5 file")
    args = parser.parse_args()
    visualize_density(args.filename)