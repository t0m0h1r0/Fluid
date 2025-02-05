import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from core.scheme import DifferenceScheme
from core.boundary import DirectionalBC

@dataclass
class PhaseFieldParams:
    epsilon: float = 0.01  # Interface thickness
    mobility: float = 1.0  # Mobility
    surface_tension: float = 0.07  # Surface tension

@dataclass
class PhaseObject:
    """Base class for phase field objects"""
    phase_name: str

@dataclass
class Layer(PhaseObject):
    """Layer object for phase field"""
    z_range: Tuple[float, float]

@dataclass
class Sphere(PhaseObject):
    """Sphere object for phase field"""
    center: Tuple[float, float, float]
    radius: float

class PhaseFieldSolver:
    def __init__(self, 
                 scheme: DifferenceScheme,
                 boundary_conditions: DirectionalBC,
                 params: PhaseFieldParams):
        self.scheme = scheme
        self.boundary_conditions = boundary_conditions
        self.params = params
        self._grid = None
        self._phase_densities = {}

    def initialize_field(self, shape: Tuple[int, ...], 
                        domain_size: Tuple[float, ...]) -> np.ndarray:
        """Initialize an empty phase field
        
        Args:
            shape: Grid dimensions (Nx, Ny, Nz)
            domain_size: Physical domain size (Lx, Ly, Lz)
            
        Returns:
            initialized phase field array
        """
        self._grid = self._create_grid(shape, domain_size)
        return np.zeros(shape)

    def set_phase_density(self, phase_name: str, density: float):
        """Register density for a phase
        
        Args:
            phase_name: Name of the phase
            density: Density value
        """
        self._phase_densities[phase_name] = density

    def add_layer(self, field: np.ndarray, layer: Layer) -> np.ndarray:
        """Add a layer to the phase field
        
        Args:
            field: Current phase field
            layer: Layer configuration
            
        Returns:
            Updated phase field
        """
        if self._grid is None:
            raise RuntimeError("Grid not initialized. Call initialize_field first.")
        
        X, Y, Z = self._grid
        z_min, z_max = layer.z_range
        mask = (Z >= z_min) & (Z < z_max)
        
        if layer.phase_name not in self._phase_densities:
            raise ValueError(f"Density not set for phase {layer.phase_name}")
            
        field[mask] = self._phase_densities[layer.phase_name]
        return field

    def add_sphere(self, field: np.ndarray, sphere: Sphere) -> np.ndarray:
        """Add a sphere to the phase field
        
        Args:
            field: Current phase field
            sphere: Sphere configuration
            
        Returns:
            Updated phase field
        """
        if self._grid is None:
            raise RuntimeError("Grid not initialized. Call initialize_field first.")
            
        X, Y, Z = self._grid
        r = np.sqrt(
            (X - sphere.center[0])**2 + 
            (Y - sphere.center[1])**2 + 
            (Z - sphere.center[2])**2
        )
        mask = r <= sphere.radius
        
        if sphere.phase_name not in self._phase_densities:
            raise ValueError(f"Density not set for phase {sphere.phase_name}")
            
        field[mask] = self._phase_densities[sphere.phase_name]
        return field

    def _create_grid(self, shape: Tuple[int, ...], 
                    domain_size: Tuple[float, ...]) -> Tuple[np.ndarray, ...]:
        """Create computational grid
        
        Args:
            shape: Grid dimensions
            domain_size: Physical domain size
        
        Returns:
            Tuple of meshgrid arrays (X, Y, Z)
        """
        x = np.linspace(0, domain_size[0], shape[0])
        y = np.linspace(0, domain_size[1], shape[1])
        z = np.linspace(0, domain_size[2], shape[2])
        return np.meshgrid(x, y, z, indexing='ij')

    def heaviside(self, phi: np.ndarray) -> np.ndarray:
        """Heaviside function approximation"""
        return 0.5 * (1.0 + np.tanh(phi / self.params.epsilon))
    
    def delta(self, phi: np.ndarray) -> np.ndarray:
        """Delta function approximation"""
        return (1.0 / (2.0 * self.params.epsilon)) * (
            1.0 - np.tanh(phi / self.params.epsilon)**2
        )
    
    def compute_chemical_potential(self, phi: np.ndarray) -> np.ndarray:
        """Compute chemical potential"""
        mu = phi * (phi**2 - 1.0) - self.params.epsilon**2 * self.compute_laplacian(phi)
        return mu
    
    def compute_laplacian(self, phi: np.ndarray) -> np.ndarray:
        """Compute Laplacian"""
        laplacian = np.zeros_like(phi)
        for axis in range(phi.ndim):
            bc = self.boundary_conditions.get_condition(axis)
            for idx in self._get_orthogonal_indices(phi.shape, axis):
                line = self._get_line(phi, axis, idx)
                d2_line = self.scheme.apply(line, bc)
                self._set_line(laplacian, axis, idx, d2_line)
        return laplacian
    
    def advance(self, phi: np.ndarray, velocity: Tuple[np.ndarray, ...], dt: float) -> np.ndarray:
        """Time evolution"""
        # Advection term
        dphi_dt = np.zeros_like(phi)
        for axis, v in enumerate(velocity):
            bc = self.boundary_conditions.get_condition(axis)
            dphi_dt -= v * self.compute_gradient(phi, axis)
        
        # Diffusion term
        mu = self.compute_chemical_potential(phi)
        dphi_dt += self.params.mobility * self.compute_laplacian(mu)
        
        return phi + dt * dphi_dt

    def compute_gradient(self, phi: np.ndarray, axis: int) -> np.ndarray:
        """Compute gradient in specified direction"""
        bc = self.boundary_conditions.get_condition(axis)
        gradient = np.zeros_like(phi)
        
        for idx in self._get_orthogonal_indices(phi.shape, axis):
            line = self._get_line(phi, axis, idx)
            grad_line = self.scheme.apply(line, bc)
            self._set_line(gradient, axis, idx, grad_line)
        
        return gradient

    def _get_orthogonal_indices(self, shape: Tuple[int, ...], axis: int):
        ranges = [range(s) for i, s in enumerate(shape) if i != axis]
        return np.array(np.meshgrid(*ranges, indexing='ij')).reshape(len(ranges), -1).T
    
    def _get_line(self, array: np.ndarray, axis: int, idx) -> np.ndarray:
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        return array[tuple(idx_list)]
    
    def _set_line(self, array: np.ndarray, axis: int, idx, values: np.ndarray):
        idx_list = list(idx)
        idx_list.insert(axis, slice(None))
        array[tuple(idx_list)] = values