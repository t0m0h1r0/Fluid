import numpy as np
import numpy.typing as npt


def heaviside(phi: npt.NDArray, epsilon: float = 1.0e-2) -> npt.NDArray:
    """正則化されたHeaviside関数"""
    return 0.5 * (1.0 + np.tanh(phi / epsilon))


def delta(phi: npt.NDArray, epsilon: float = 1.0e-2) -> npt.NDArray:
    """正則化されたDelta関数"""
    return 0.5 / epsilon * (1.0 - np.tanh(phi / epsilon) ** 2)


def compute_curvature(phi: npt.NDArray, dx: float) -> npt.NDArray:
    """界面の曲率を計算"""
    grad = np.array(np.gradient(phi, dx))
    grad_norm = np.sqrt(np.sum(grad**2, axis=0))
    grad_norm = np.maximum(grad_norm, 1e-10)

    return sum(np.gradient(grad[i] / grad_norm, dx, axis=i) for i in range(phi.ndim))


def compute_volume(phi: npt.NDArray, dx: float) -> float:
    """Level Set関数から体積を計算"""
    return float(np.sum(heaviside(phi)) * dx**phi.ndim)


def compute_area(phi: npt.NDArray, dx: float) -> float:
    """Level Set関数から界面の面積を計算"""
    return float(np.sum(delta(phi)) * dx**phi.ndim)


def extend_velocity(
    velocity: npt.NDArray, phi: npt.NDArray, dx: float, n_steps: int = 5
) -> npt.NDArray:
    """界面に沿って速度場を延長"""
    result = velocity.copy()

    for _ in range(n_steps):
        # 法線ベクトルを計算
        grad = np.array(np.gradient(phi, dx))
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))
        grad_norm = np.maximum(grad_norm, 1e-10)
        normal = grad / grad_norm

        # 速度勾配を計算
        vel_grad = np.array(np.gradient(result, dx))

        # 法線方向の速度勾配を解消
        result -= sum(normal[i] * vel_grad[i] for i in range(phi.ndim))

    return result
