import numpy as np
from .base import BoundaryCondition
from ..field.base import Field

class PeriodicBoundary(BoundaryCondition):
    def apply(self, field: Field) -> Field:
        data = field.data
        if isinstance(data, list):  # ベクトル場
            for i in range(len(data)):
                data[i] = self._apply_periodic(data[i])
        else:  # スカラー場
            data = self._apply_periodic(data)
        field.data = data
        return field

    def _apply_periodic(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        for axis in range(3):
            result = np.concatenate([result[-1:], result, result[:1]], axis=axis)
        return result

    def get_ghost_points(self, field: Field) -> List[np.ndarray]:
        data = field.data
        if isinstance(data, list):
            return [self._get_periodic_ghosts(d) for d in data]
        return [self._get_periodic_ghosts(data)]

    def _get_periodic_ghosts(self, data: np.ndarray) -> np.ndarray:
        ghosts = []
        for axis in range(3):
            ghosts.extend([data[-1:], data[:1]])
        return np.array(ghosts)

    def validate(self, field: Field) -> bool:
        return True

class DirichletBoundary(BoundaryCondition):
    def __init__(self, value: float):
        self.value = value

    def apply(self, field: Field) -> Field:
        data = field.data
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = self._apply_dirichlet(data[i])
        else:
            data = self._apply_dirichlet(data)
        field.data = data
        return field

    def _apply_dirichlet(self, data: np.ndarray) -> np.ndarray:
        result = data.copy()
        result[0] = self.value
        result[-1] = self.value
        return result

    def get_ghost_points(self, field: Field) -> List[np.ndarray]:
        return [np.full((2,), self.value)]

    def validate(self, field: Field) -> bool:
        data = field.data
        if isinstance(data, list):
            return all(self._validate_dirichlet(d) for d in data)
        return self._validate_dirichlet(data)

    def _validate_dirichlet(self, data: np.ndarray) -> bool:
        return (np.allclose(data[0], self.value) and 
                np.allclose(data[-1], self.value))

class NeumannBoundary(BoundaryCondition):
    def __init__(self, gradient: float = 0.0):
        self.gradient = gradient

    def apply(self, field: Field) -> Field:
        data = field.data
        dx = field.dx[0]  # 簡単のため1方向のみ考慮
        
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = self._apply_neumann(data[i], dx)
        else:
            data = self._apply_neumann(data, dx)
        
        field.data = data
        return field

    def _apply_neumann(self, data: np.ndarray, dx: float) -> np.ndarray:
        result = data.copy()
        result[0] = result[1] - self.gradient * dx
        result[-1] = result[-2] + self.gradient * dx
        return result

    def get_ghost_points(self, field: Field) -> List[np.ndarray]:
        data = field.data
        dx = field.dx[0]
        
        if isinstance(data, list):
            return [self._get_neumann_ghosts(d, dx) for d in data]
        return [self._get_neumann_ghosts(data, dx)]

    def _get_neumann_ghosts(self, data: np.ndarray, dx: float) -> np.ndarray:
        ghost_minus = data[1] - self.gradient * dx
        ghost_plus = data[-2] + self.gradient * dx
        return np.array([ghost_minus, ghost_plus])

    def validate(self, field: Field) -> bool:
        data = field.data
        dx = field.dx[0]
        
        if isinstance(data, list):
            return all(self._validate_neumann(d, dx) for d in data)
        return self._validate_neumann(data, dx)

    def _validate_neumann(self, data: np.ndarray, dx: float) -> bool:
        grad_minus = (data[1] - data[0]) / dx
        grad_plus = (data[-1] - data[-2]) / dx
        return (np.allclose(grad_minus, self.gradient) and 
                np.allclose(grad_plus, self.gradient))