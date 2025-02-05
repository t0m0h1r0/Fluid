from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple

class SpatialDiscretization(ABC):
    @abstractmethod
    def compute_gradient(self, field: np.ndarray, dx: float, axis: int) -> np.ndarray:
        pass

    @abstractmethod
    def compute_divergence(self, vector_field: List[np.ndarray], dx: Tuple[float, ...]) -> np.ndarray:
        pass

    @abstractmethod
    def compute_laplacian(self, field: np.ndarray, dx: float) -> np.ndarray:
        pass

class CentralDifference(SpatialDiscretization):
    """中心差分法"""
    def compute_gradient(self, field: np.ndarray, dx: float, axis: int) -> np.ndarray:
        return np.gradient(field, dx, axis=axis)

    def compute_divergence(self, vector_field: List[np.ndarray], dx: Tuple[float, ...]) -> np.ndarray:
        div = np.zeros_like(vector_field[0])
        for i, v in enumerate(vector_field):
            div += np.gradient(v, dx[i], axis=i)
        return div

    def compute_laplacian(self, field: np.ndarray, dx: float) -> np.ndarray:
        laplacian = np.zeros_like(field)
        for i in range(field.ndim):
            laplacian += np.gradient(np.gradient(field, dx, axis=i), dx, axis=i)
        return laplacian

class WENO(SpatialDiscretization):
    """5次精度WENO法"""
    def __init__(self):
        self.epsilon = 1e-6
        self.optimal_weights = np.array([0.1, 0.6, 0.3])

    def compute_gradient(self, field: np.ndarray, dx: float, axis: int) -> np.ndarray:
        return self._weno_derivative(field, dx, axis)

    def compute_divergence(self, vector_field: List[np.ndarray], dx: Tuple[float, ...]) -> np.ndarray:
        div = np.zeros_like(vector_field[0])
        for i, v in enumerate(vector_field):
            div += self._weno_derivative(v, dx[i], i)
        return div

    def compute_laplacian(self, field: np.ndarray, dx: float) -> np.ndarray:
        laplacian = np.zeros_like(field)
        for i in range(field.ndim):
            grad = self._weno_derivative(field, dx, i)
            laplacian += self._weno_derivative(grad, dx, i)
        return laplacian

    def _weno_derivative(self, field: np.ndarray, dx: float, axis: int) -> np.ndarray:
        # シフトされた値の取得
        v = [np.roll(field, i, axis=axis) for i in range(-2, 3)]
        
        # 3つのステンシルでの近似
        s0 = (13/12) * (v[0] - 2*v[1] + v[2])**2 + \
             (1/4) * (v[0] - 4*v[1] + 3*v[2])**2
        s1 = (13/12) * (v[1] - 2*v[2] + v[3])**2 + \
             (1/4) * (v[1] - v[3])**2
        s2 = (13/12) * (v[2] - 2*v[3] + v[4])**2 + \
             (1/4) * (3*v[2] - 4*v[3] + v[4])**2
        
        # 非線形重み
        alpha = self.optimal_weights / (self.epsilon + np.array([s0, s1, s2]))**2
        omega = alpha / alpha.sum(axis=0)
        
        # 各ステンシルでの補間
        p0 = (1/6) * (2*v[0] - 7*v[1] + 11*v[2])
        p1 = (1/6) * (-v[1] + 5*v[2] + 2*v[3])
        p2 = (1/6) * (2*v[2] + 5*v[3] - v[4])
        
        # 最終的な近似
        result = omega[0]*p0 + omega[1]*p1 + omega[2]*p2
        return result / dx

class CompactScheme(SpatialDiscretization):
    """4次精度コンパクトスキーム"""
    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha

    def compute_gradient(self, field: np.ndarray, dx: float, axis: int) -> np.ndarray:
        n = field.shape[axis]
        
        # トリディアゴナル行列の係数
        a = np.full(n-1, self.alpha)
        b = np.ones(n)
        c = np.full(n-1, self.alpha)
        
        # 右辺の計算
        r = np.zeros(n)
        slices = [slice(None)] * field.ndim
        for i in range(1, n-1):
            slices[axis] = i
            r[i] = (field[tuple(slices_plus(slices, axis, 1))] - 
                   field[tuple(slices_minus(slices, axis, 1))]) / (2*dx)
        
        # 境界条件
        r[0] = (-1.5*field[tuple(slices_at(slices, axis, 0))] + 
                2.0*field[tuple(slices_at(slices, axis, 1))] - 
                0.5*field[tuple(slices_at(slices, axis, 2))]) / dx
        r[-1] = (1.5*field[tuple(slices_at(slices, axis, -1))] - 
                2.0*field[tuple(slices_at(slices, axis, -2))] + 
                0.5*field[tuple(slices_at(slices, axis, -3))]) / dx
        
        # トリディアゴナルシステムを解く
        return self._solve_tridiagonal(a, b, c, r)

    def compute_divergence(self, vector_field: List[np.ndarray], dx: Tuple[float, ...]) -> np.ndarray:
        div = np.zeros_like(vector_field[0])
        for i, v in enumerate(vector_field):
            div += self.compute_gradient(v, dx[i], i)
        return div

    def compute_laplacian(self, field: np.ndarray, dx: float) -> np.ndarray:
        laplacian = np.zeros_like(field)
        for i in range(field.ndim):
            grad = self.compute_gradient(field, dx, i)
            laplacian += self.compute_gradient(grad, dx, i)
        return laplacian

    def _solve_tridiagonal(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, 
                          r: np.ndarray) -> np.ndarray:
        """トーマスアルゴリズムによるトリディアゴナル行列の解法"""
        n = len(r)
        u = np.zeros(n)
        gam = np.zeros(n)
        
        bet = b[0]
        u[0] = r[0] / bet
        
        for i in range(1, n):
            gam[i] = c[i-1] / bet
            bet = b[i] - a[i-1] * gam[i]
            u[i] = (r[i] - a[i-1] * u[i-1]) / bet
        
        for i in range(n-2, -1, -1):
            u[i] -= gam[i+1] * u[i+1]
        
        return u

def slices_plus(slices: List[slice], axis: int, offset: int) -> Tuple[slice, ...]:
    result = slices.copy()
    result[axis] = slice(offset, None)
    return tuple(result)

def slices_minus(slices: List[slice], axis: int, offset: int) -> Tuple[slice, ...]:
    result = slices.copy()
    result[axis] = slice(None, -offset)
    return tuple(result)

def slices_at(slices: List[slice], axis: int, index: int) -> Tuple[slice, ...]:
    result = slices.copy()
    result[axis] = index
    return tuple(result)