"""Level Set法のユーティリティ関数を提供するモジュール"""

import numpy as np


def heaviside(phi: np.ndarray, epsilon: float = 1.0e-2) -> np.ndarray:
    """正則化されたHeaviside関数

    Args:
        phi: Level Set関数
        epsilon: 界面の厚さ

    Returns:
        正則化されたHeaviside関数の値
    """
    return 0.5 * (1.0 + np.tanh(phi / epsilon))


def delta(phi: np.ndarray, epsilon: float = 1.0e-2) -> np.ndarray:
    """正則化されたDelta関数

    Args:
        phi: Level Set関数
        epsilon: 界面の厚さ

    Returns:
        正則化されたDelta関数の値
    """
    return 0.5 / epsilon * (1.0 - np.tanh(phi / epsilon) ** 2)


def compute_curvature(phi: np.ndarray, dx: float) -> np.ndarray:
    """界面の曲率を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔

    Returns:
        界面の曲率
    """
    # 勾配を計算
    grad = np.array(np.gradient(phi, dx))
    grad_norm = np.sqrt(np.sum(grad**2, axis=0))
    grad_norm = np.maximum(grad_norm, 1e-10)  # ゼロ除算を防ぐ

    # 正規化された勾配
    grad_normalized = grad / grad_norm

    # 発散を計算して曲率を得る
    return sum(np.gradient(grad_normalized[i], dx, axis=i) for i in range(phi.ndim))


def compute_volume(phi: np.ndarray, dx: float) -> float:
    """Level Set関数から体積を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔

    Returns:
        計算された体積
    """
    return float(np.sum(heaviside(phi)) * dx**phi.ndim)


def compute_area(phi: np.ndarray, dx: float) -> float:
    """Level Set関数から界面の面積を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔

    Returns:
        計算された界面の面積
    """
    return float(np.sum(delta(phi)) * dx**phi.ndim)


def compute_interface_gradient(phi: np.ndarray, dx: float) -> np.ndarray:
    """界面の法線ベクトルを計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔

    Returns:
        界面の法線ベクトル
    """
    # 勾配を計算
    grad = np.array(np.gradient(phi, dx))

    # 勾配の大きさを計算
    grad_norm = np.sqrt(np.sum(grad**2, axis=0))
    grad_norm = np.maximum(grad_norm, 1e-10)  # ゼロ除算を防ぐ

    # 正規化された勾配（法線ベクトル）
    return grad / grad_norm


def extend_velocity(
    velocity: np.ndarray, phi: np.ndarray, dx: float, n_steps: int = 5
) -> np.ndarray:
    """界面の法線方向に速度場を拡張

    Args:
        velocity: 速度場
        phi: Level Set関数
        dx: グリッド間隔
        n_steps: 拡張のステップ数

    Returns:
        拡張された速度場
    """
    result = velocity.copy()
    dt = 0.5 * dx  # 仮想時間の時間刻み幅

    for _ in range(n_steps):
        # 法線ベクトルを計算
        grad = np.array(np.gradient(phi, dx))
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))
        grad_norm = np.maximum(grad_norm, 1e-10)
        normal = grad / grad_norm

        # 速度勾配を計算
        vel_grad = np.array(np.gradient(result, dx))

        # 法線方向の速度勾配を解消
        result -= dt * sum(normal[i] * vel_grad[i] for i in range(phi.ndim))

    return result
