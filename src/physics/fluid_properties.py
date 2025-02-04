import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class FluidProperty:
    """Data class for fluid properties"""
    name: str
    density: float  # kg/m^3
    viscosity: float  # Pa.s
    
class FluidProperties:
    """Manages properties of multiple fluids and their distribution in space"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize fluid properties manager
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing:
            - Grid parameters (nx, ny, nz)
            - Fluid definitions
        """
        # Grid parameters
        self.nx = config['nx']
        self.ny = config['ny']
        self.nz = config['nz']
        
        # Initialize property fields
        self.density = np.zeros((self.nx, self.ny, self.nz))
        self.viscosity = np.zeros((self.nx, self.ny, self.nz))
        self.phase_indicator = np.zeros((self.nx, self.ny, self.nz))
        
        # Store fluid definitions
        self.fluids: Dict[str, FluidProperty] = {}
        self._register_fluids(config.get('fluids', {}))
    
    def _register_fluids(self, fluid_configs: Dict[str, Dict[str, float]]) -> None:
        """Register fluid properties from configuration"""
        for name, props in fluid_configs.items():
            self.fluids[name] = FluidProperty(
                name=name,
                density=props['density'],
                viscosity=props['viscosity']
            )
    
    def add_fluid(self, name: str, density: float, viscosity: float) -> None:
        """Add a new fluid definition"""
        self.fluids[name] = FluidProperty(name, density, viscosity)
    
    def _smoothed_heaviside(self, phi: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Compute smoothed Heaviside function
        
        Parameters
        ----------
        phi : ndarray
            Distance function
        epsilon : float
            Transition region thickness
        """
        heaviside = np.zeros_like(phi)
        
        # Smoothed transition region
        region1 = np.abs(phi) <= epsilon
        heaviside[phi > epsilon] = 1.0
        heaviside[region1] = 0.5 * (1 + phi[region1]/epsilon + 
                                   np.sin(np.pi * phi[region1]/epsilon)/np.pi)
        
        return heaviside
    
    def set_two_fluid_interface(self, 
                              fluid1: str, 
                              fluid2: str,
                              distance_function: np.ndarray,
                              epsilon: Optional[float] = None) -> None:
        """
        Set interface between two fluids using a distance function
        
        Parameters
        ----------
        fluid1 : str
            Name of first fluid
        fluid2 : str
            Name of second fluid
        distance_function : ndarray
            Signed distance function (positive in fluid1, negative in fluid2)
        epsilon : float, optional
            Transition region thickness. If None, uses grid spacing
        """
        if fluid1 not in self.fluids or fluid2 not in self.fluids:
            raise ValueError("Fluid not found")
            
        if epsilon is None:
            epsilon = max(self.nx, self.ny, self.nz) / 100.0
            
        # Compute smooth Heaviside function
        H = self._smoothed_heaviside(distance_function, epsilon)
        
        # Set properties using mixture
        prop1 = self.fluids[fluid1]
        prop2 = self.fluids[fluid2]
        
        self.density = prop2.density + (prop1.density - prop2.density) * H
        self.viscosity = prop2.viscosity + (prop1.viscosity - prop2.viscosity) * H
        self.phase_indicator = H
    
    def get_density(self) -> np.ndarray:
        """Get density field"""
        return self.density.copy()
    
    def get_viscosity(self) -> np.ndarray:
        """Get viscosity field"""
        return self.viscosity.copy()
    
    def get_phase_indicator(self) -> np.ndarray:
        """Get phase indicator field"""
        return self.phase_indicator.copy()

class TwoLayerProperties(FluidProperties):
    """Specialized class for two-layer fluid configuration"""
    
    def set_bubble_and_layer(self,
                            bubble_center: np.array,
                            bubble_radius: float,
                            layer_height: float,
                            fluid_bubble: str,
                            fluid_ambient: str,
                            grid_coordinates: tuple) -> None:
        """
        Set up a bubble and layer configuration
        
        Parameters
        ----------
        bubble_center : array
            Center coordinates of bubble [x, y, z]
        bubble_radius : float
            Radius of bubble
        layer_height : float
            Height of top layer
        fluid_bubble : str
            Name of fluid in bubble
        fluid_ambient : str
            Name of ambient fluid
        grid_coordinates : tuple
            Tuple of coordinate meshgrids (X, Y, Z)
        """
        X, Y, Z = grid_coordinates
        
        # Compute distance functions
        bubble_distance = np.sqrt(
            (X - bubble_center[0])**2 + 
            (Y - bubble_center[1])**2 + 
            (Z - bubble_center[2])**2
        ) - bubble_radius
        
        layer_distance = -(Z - layer_height)
        
        # Combined distance function (negative inside any fluid1 region)
        distance = np.minimum(bubble_distance, layer_distance)
        
        # Set properties using parent method
        self.set_two_fluid_interface(fluid_bubble, fluid_ambient, -distance)