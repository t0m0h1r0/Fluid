from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional
from core.field.scalar_field import ScalarField
from core.field.vector_field import VectorField

class Stabilizer(ABC):
    """数値安定化の基底クラス"""
    @abstractmethod
    def apply(self, field: ScalarField | VectorField) -> None:
        pass

class ArtificialViscosity(Stabilizer):
    """人工粘性による安定化"""
    def __init__(self, coefficient: float = 0.1):
        self.coefficient = coefficient

    def apply(self, field: ScalarField | VectorField) -> None:
        if isinstance(field, ScalarField):
            self._apply_scalar(field)
        else:
            self._apply_vector(field)

    def _apply_scalar(self, field: ScalarField) -> None:
        dx = field.dx
        data = field.data
        
        # 2次の人工粘性項を追加
        laplacian = np.zeros_like(data)
        for i in range(3):
            laplacian += np.gradient(np.gradient(data, dx[i], axis=i), dx[i], axis=i)
        
        field.data += self.coefficient * laplacian

    def _apply_vector(self, field: VectorField) -> None:
        for component in field.data:
            temp_field = ScalarField(field.metadata)
            temp_field.data = component
            self._apply_scalar(temp_field)
            component[:] = temp_field.data

class FluxLimiter(Stabilizer):
    """フラックスリミッターによる安定化"""
    def __init__(self, limiter_type: str = 'minmod'):
        self.limiter_type = limiter_type
        self.limiters = {
            'minmod': self._minmod,
            'superbee': self._superbee,
            'van_leer': self._van_leer
        }

    def apply(self, field: ScalarField | VectorField) -> None:
        if isinstance(field, ScalarField):
            self._apply_scalar(field)
        else:
            self._apply_vector(field)

    def _apply_scalar(self, field: ScalarField) -> None:
        dx = field.dx
        data = field.data
        limiter = self.limiters[self.limiter_type]
        
        # 各方向についてリミッターを適用
        for i in range(3):
            grad = np.gradient(data, dx[i], axis=i)
            r = self._compute_ratio(grad, axis=i)
            phi = limiter(r)
            grad *= phi
            data += dx[i] * grad

    def _apply_vector(self, field: VectorField) -> None:
        for component in field.data:
            temp_field = ScalarField(field.metadata)
            temp_field.data = component
            self._apply_scalar(temp_field)
            component[:] = temp_field.data

    def _compute_ratio(self, grad: np.ndarray, axis: int) -> np.ndarray:
        """勾配比の計算"""
        grad_plus = np.roll(grad, -1, axis=axis)
        grad_minus = np.roll(grad, 1, axis=axis)
        
        # ゼロ除算を防ぐ
        eps = 1e-10
        ratio = np.where(
            np.abs(grad) > eps,
            grad_plus / (grad + eps),
            1.0
        )
        return ratio

    def _minmod(self, r: np.ndarray) -> np.ndarray:
        """minmodリミッター"""
        return np.maximum(0.0, np.minimum(1.0, r))

    def _superbee(self, r: np.ndarray) -> np.ndarray:
        """superbeeリミッター"""
        return np.maximum(0.0, np.maximum(
            np.minimum(2.0*r, 1.0),
            np.minimum(r, 2.0)
        ))

    def _van_leer(self, r: np.ndarray) -> np.ndarray:
        """van Leerリミッター"""
        return (r + np.abs(r)) / (1.0 + np.abs(r))

class Filtering(Stabilizer):
    """フィルタリングによる安定化"""
    def __init__(self, filter_width: int = 3, strength: float = 0.1):
        self.filter_width = filter_width
        self.strength = strength

    def apply(self, field: ScalarField | VectorField) -> None:
        if isinstance(field, ScalarField):
            self._apply_scalar(field)
        else:
            self._apply_vector(field)

    def _apply_scalar(self, field: ScalarField) -> None:
        data = field.data
        filtered = np.zeros_like(data)
        
        # 移動平均フィルター
        for i in range(3):
            kernel = np.ones(self.filter_width) / self.filter_width
            filtered += np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode='same'),
                i, data
            )
        
        field.data = (1.0 - self.strength) * data + self.strength * filtered / 3.0

    def _apply_vector(self, field: VectorField) -> None:
        for component in field.data:
            temp_field = ScalarField(field.metadata)
            temp_field.data = component
            self._apply_scalar(temp_field)
            component[:] = temp_field.data

class StabilizerComposite(Stabilizer):
    """複数の安定化手法の組み合わせ"""
    def __init__(self, stabilizers: Optional[List[Stabilizer]] = None):
        self.stabilizers = stabilizers or []

    def add_stabilizer(self, stabilizer: Stabilizer) -> None:
        self.stabilizers.append(stabilizer)

    def apply(self, field: ScalarField | VectorField) -> None:
        for stabilizer in self.stabilizers:
            stabilizer.apply(field)