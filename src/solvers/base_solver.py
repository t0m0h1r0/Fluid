from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Dict, Any

class BaseSolver(ABC):
    """
    Abstract base class for numerical solvers
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize solver with configuration

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing grid and solver parameters
        """
        # Grid parameters
        self.nx = config['nx']
        self.ny = config['ny']
        self.nz = config['nz']
        self.dx = config['dx']
        self.dy = config['dy']
        self.dz = config['dz']
        
        # Time stepping parameters
        self.dt = config['dt']
        self.t = 0.0
        
        # Initialize fields
        self.u = np.zeros((self.nx, self.ny, self.nz))
        self.v = np.zeros((self.nx, self.ny, self.nz))
        self.w = np.zeros((self.nx, self.ny, self.nz))
        self.p = np.zeros((self.nx, self.ny, self.nz))
        
        # Store configuration
        self.config = config
    
    @abstractmethod
    def step(self) -> None:
        """Advance solution by one time step"""
        pass
    
    @abstractmethod
    def solve(self, end_time: float) -> None:
        """
        Solve until specified end time
        
        Parameters
        ----------
        end_time : float
            Time to solve until
        """
        pass
    
    def set_initial_condition(self, initial_condition: Dict[str, np.ndarray]) -> None:
        """
        Set initial condition for all fields
        
        Parameters
        ----------
        initial_condition : Dict[str, np.ndarray]
            Dictionary containing initial values for fields
        """
        for field_name, value in initial_condition.items():
            if hasattr(self, field_name):
                setattr(self, field_name, value.copy())
    
    @abstractmethod
    def apply_boundary_condition(self) -> None:
        """Apply boundary conditions to all fields"""
        pass
    
    def get_field(self, field_name: str) -> np.ndarray:
        """
        Get copy of field data
        
        Parameters
        ----------
        field_name : str
            Name of field to retrieve
            
        Returns
        -------
        np.ndarray
            Copy of field data
        """
        if hasattr(self, field_name):
            return getattr(self, field_name).copy()
        raise ValueError(f"Field {field_name} not found")
    
    def set_field(self, field_name: str, value: np.ndarray) -> None:
        """
        Set field data
        
        Parameters
        ----------
        field_name : str
            Name of field to set
        value : np.ndarray
            Value to set field to
        """
        if hasattr(self, field_name):
            setattr(self, field_name, value.copy())
        else:
            raise ValueError(f"Field {field_name} not found")
    
    def get_time(self) -> float:
        """Get current simulation time"""
        return self.t
    
    @abstractmethod
    def compute_derivatives(self, field: np.ndarray, 
                          direction: str) -> np.ndarray:
        """
        Compute spatial derivatives
        
        Parameters
        ----------
        field : np.ndarray
            Field to compute derivatives of
        direction : str
            Direction to compute derivative in ('x', 'y', or 'z')
            
        Returns
        -------
        np.ndarray
            Computed derivative
        """
        pass