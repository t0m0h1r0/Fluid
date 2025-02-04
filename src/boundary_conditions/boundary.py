from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Optional

class BoundaryType(Enum):
    """Enumeration of boundary condition types"""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    PERIODIC = "periodic"

class BoundaryLocation(Enum):
    """Enumeration of boundary locations"""
    LEFT = "left"      # x = 0
    RIGHT = "right"    # x = Lx
    BOTTOM = "bottom"  # y = 0
    TOP = "top"       # y = Ly
    FRONT = "front"   # z = 0
    BACK = "back"     # z = Lz

class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions"""
    
    def __init__(self, location: BoundaryLocation, bc_type: BoundaryType):
        self.location = location
        self.bc_type = bc_type
    
    @abstractmethod
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply boundary condition to field"""
        pass

class NeumannBC(BoundaryCondition):
    """Neumann boundary condition implementation"""
    
    def __init__(self, location: BoundaryLocation, gradient: float = 0.0):
        super().__init__(location, BoundaryType.NEUMANN)
        self.gradient = gradient
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply Neumann boundary condition"""
        if self.location == BoundaryLocation.LEFT:
            field[0, :, :] = field[1, :, :] - self.gradient
        elif self.location == BoundaryLocation.RIGHT:
            field[-1, :, :] = field[-2, :, :] + self.gradient
        elif self.location == BoundaryLocation.BOTTOM:
            field[:, 0, :] = field[:, 1, :] - self.gradient
        elif self.location == BoundaryLocation.TOP:
            field[:, -1, :] = field[:, -2, :] + self.gradient
        elif self.location == BoundaryLocation.FRONT:
            field[:, :, 0] = field[:, :, 1] - self.gradient
        elif self.location == BoundaryLocation.BACK:
            field[:, :, -1] = field[:, :, -2] + self.gradient
        
        return field

class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition implementation"""
    
    def __init__(self, location: BoundaryLocation, value: float):
        super().__init__(location, BoundaryType.DIRICHLET)
        self.value = value
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply Dirichlet boundary condition"""
        if self.location == BoundaryLocation.LEFT:
            field[0, :, :] = self.value
        elif self.location == BoundaryLocation.RIGHT:
            field[-1, :, :] = self.value
        elif self.location == BoundaryLocation.BOTTOM:
            field[:, 0, :] = self.value
        elif self.location == BoundaryLocation.TOP:
            field[:, -1, :] = self.value
        elif self.location == BoundaryLocation.FRONT:
            field[:, :, 0] = self.value
        elif self.location == BoundaryLocation.BACK:
            field[:, :, -1] = self.value
        
        return field

class PeriodicBC(BoundaryCondition):
    """Periodic boundary condition implementation"""
    
    def __init__(self, location: BoundaryLocation):
        super().__init__(location, BoundaryType.PERIODIC)
    
    def apply(self, field: np.ndarray) -> np.ndarray:
        """Apply periodic boundary condition"""
        if self.location in [BoundaryLocation.LEFT, BoundaryLocation.RIGHT]:
            field[0, :, :] = field[-2, :, :]
            field[-1, :, :] = field[1, :, :]
        elif self.location in [BoundaryLocation.BOTTOM, BoundaryLocation.TOP]:
            field[:, 0, :] = field[:, -2, :]
            field[:, -1, :] = field[:, 1, :]
        elif self.location in [BoundaryLocation.FRONT, BoundaryLocation.BACK]:
            field[:, :, 0] = field[:, :, -2]
            field[:, :, -1] = field[:, :, 1]
        
        return field

class BoundaryManager:
    """Manages boundary conditions for all fields"""
    
    def __init__(self):
        self.boundary_conditions: Dict[str, Dict[BoundaryLocation, BoundaryCondition]] = {}
    
    def add_boundary_condition(self, 
                             field_name: str,
                             location: BoundaryLocation,
                             condition: BoundaryCondition) -> None:
        """Add boundary condition for a field at a specific location"""
        if field_name not in self.boundary_conditions:
            self.boundary_conditions[field_name] = {}
        
        self.boundary_conditions[field_name][location] = condition
    
    def apply_boundary_conditions(self, fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all boundary conditions to fields"""
        for field_name, field in fields.items():
            if field_name in self.boundary_conditions:
                for bc in self.boundary_conditions[field_name].values():
                    field = bc.apply(field)
                fields[field_name] = field
        
        return fields

class NSBoundaryManager(BoundaryManager):
    """Specialized boundary manager for Navier-Stokes equations"""
    
    def apply_ns_boundary_conditions(self, 
                                   u: np.ndarray,
                                   v: np.ndarray,
                                   w: np.ndarray,
                                   p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply boundary conditions to NS fields"""
        fields = {
            'u': u,
            'v': v,
            'w': w,
            'p': p
        }
        
        fields = self.apply_boundary_conditions(fields)
        
        return fields['u'], fields['v'], fields['w'], fields['p']
    
    def set_all_neumann(self) -> None:
        """Set Neumann boundary conditions for all fields and boundaries"""
        for field in ['u', 'v', 'w', 'p']:
            for location in BoundaryLocation:
                self.add_boundary_condition(
                    field,
                    location,
                    NeumannBC(location)
                )
    
    def set_lid_driven_cavity(self) -> None:
        """Set boundary conditions for lid-driven cavity problem"""
        # No-slip conditions for velocity
        for field in ['u', 'v', 'w']:
            for location in BoundaryLocation:
                if location == BoundaryLocation.TOP and field == 'u':
                    # Moving lid
                    self.add_boundary_condition(field, location, DirichletBC(location, 1.0))
                else:
                    # No-slip walls
                    self.add_boundary_condition(field, location, DirichletBC(location, 0.0))
        
        # Neumann condition for pressure
        for location in BoundaryLocation:
            self.add_boundary_condition('p', location, NeumannBC(location))