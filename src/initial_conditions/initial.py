import numpy as np
from typing import Dict, Any, Tuple

class TwoLayerInitialCondition:
    """Initial condition for two-layer flow with bubble"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize two-layer initial condition
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing grid and fluid properties
        """
        # Grid parameters
        self.nx = config['nx']
        self.ny = config['ny']
        self.nz = config['nz']
        self.dx = config['dx']
        self.dy = config['dy']
        self.dz = config['dz']
        
        # Domain size
        self.Lx = self.nx * self.dx
        self.Ly = self.ny * self.dy
        self.Lz = self.nz * self.dz
        
        # Fluid properties
        self.rho_water = config['fluids']['water']['density']
        self.rho_nitrogen = config['fluids']['nitrogen']['density']
        self.mu_water = config['fluids']['water']['viscosity']
        self.mu_nitrogen = config['fluids']['nitrogen']['viscosity']
        
        # Bubble parameters
        self.bubble_center = config.get('bubble_center', [0.5, 0.5, 0.2])
        self.bubble_radius = config.get('bubble_radius', 0.1)
        self.layer_height = config.get('layer_height', 1.8)
        
        # Physical parameters
        self.gravity = config.get('gravity', 9.81)
        
    def initialize_fields(self) -> Dict[str, np.ndarray]:
        """
        Initialize all fields
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing initialized fields
        """
        # Create grid
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        z = np.linspace(0, self.Lz, self.nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Initialize phase field
        phase = self._compute_phase_field(X, Y, Z)
        
        # Initialize density and viscosity fields
        rho = self.rho_water + (self.rho_nitrogen - self.rho_water) * phase
        mu = self.mu_water + (self.mu_nitrogen - self.mu_water) * phase
        
        # Ensure no zero density values
        rho = np.maximum(rho, 1e-6)
        
        # Initialize velocity fields to zero
        u = np.zeros_like(X)
        v = np.zeros_like(X)
        w = np.zeros_like(X)
        
        # Initialize pressure field with hydrostatic pressure
        p = self._compute_hydrostatic_pressure(Z, rho)
        
        return {
            'u': u,
            'v': v,
            'w': w,
            'p': p,
            'rho': rho,
            'mu': mu,
            'phase': phase
        }
    
    def _compute_phase_field(self, X: np.ndarray, Y: np.ndarray, 
                           Z: np.ndarray) -> np.ndarray:
        """
        Compute phase field (0 for water, 1 for nitrogen)
        """
        # Distance function for bubble
        bubble_distance = np.sqrt(
            (X - self.bubble_center[0])**2 + 
            (Y - self.bubble_center[1])**2 + 
            (Z - self.bubble_center[2])**2
        ) - self.bubble_radius
        
        # Distance function for top layer
        layer_distance = -(Z - self.layer_height)
        
        # Combine with smooth transition
        epsilon = max(self.dx, self.dy, self.dz) * 2
        
        # Distance function (negative inside nitrogen regions)
        distance = np.minimum(bubble_distance, layer_distance)
        
        # Smoothed Heaviside function
        phase = self._smoothed_heaviside(-distance, epsilon)
        
        return phase
    
    def _smoothed_heaviside(self, phi: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Compute smoothed Heaviside function
        """
        heaviside = np.zeros_like(phi)
        
        # Transition region
        mask = np.abs(phi) <= epsilon
        heaviside[phi > epsilon] = 1.0
        heaviside[mask] = 0.5 * (1 + phi[mask]/epsilon + 
                                np.sin(np.pi * phi[mask]/epsilon)/np.pi)
        
        return heaviside
    
    def _compute_hydrostatic_pressure(self, Z: np.ndarray, 
                                    rho: np.ndarray) -> np.ndarray:
        """
        Compute hydrostatic pressure field
        """
        # Compute hydrostatic pressure (p = ρgh from top)
        max_height = np.max(Z)
        return rho * self.gravity * (max_height - Z)