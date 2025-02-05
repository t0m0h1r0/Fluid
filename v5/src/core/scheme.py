from abc import ABC, abstractmethod
import numpy as np

class StencilOperator:
    def __init__(self, points, coefficients):
        self.points = np.array(points)
        self.coefficients = np.array(coefficients)

class BoundaryCondition(ABC):
    @abstractmethod
    def get_stencil_operator(self) -> StencilOperator:
        pass
    
    @abstractmethod
    def apply_to_field(self, field: np.ndarray) -> np.ndarray:
        pass

class DifferenceScheme(ABC):
    @abstractmethod
    def create_operator(self, size: int, boundary_condition: BoundaryCondition):
        pass
    
    @abstractmethod
    def apply(self, field: np.ndarray, boundary_condition: BoundaryCondition):
        pass