"""Level Set法で使用するユーティリティ関数を提供するモジュール

このモジュールは、Level Set法で必要となるHeaviside関数やDelta関数、
符号付き距離関数の再初期化などのユーティリティ関数を提供します。
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def heaviside(phi: np.ndarray, epsilon: float = 1.0e-2) -> np.ndarray:
    """正則化されたHeaviside関数

    Args:
        phi: Level Set関数
        epsilon: 正則化パラメータ（界面の厚さ）

    Returns:
        正則化されたHeaviside関数の値
    """
    return 0.5 * (1.0 + np.tanh(phi / epsilon))


def delta(phi: np.ndarray, epsilon: float = 1.0e-2) -> np.ndarray:
    """正則化されたDelta関数

    Args:
        phi: Level Set関数
        epsilon: 正則化パラメータ（界面の厚さ）

    Returns:
        正則化されたDelta関数の値
    """
    return 0.5 / epsilon * (1.0 - np.tanh(phi / epsilon) ** 2)


def compute_curvature(phi: np.ndarray, dx: float) -> np.ndarray:
    """Level Set関数から曲率を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔

    Returns:
        界面の曲率
    """
    # 各方向の勾配を計算
    grad = np.array(np.gradient(phi, dx))
    grad_norm = np.sqrt(np.sum(grad**2, axis=0))
    grad_norm = np.maximum(grad_norm, 1e-10)  # ゼロ除算を防ぐ

    # 正規化された勾配
    grad_normalized = grad / grad_norm

    # 発散を計算して曲率を得る
    return sum(np.gradient(grad_normalized[i], dx, axis=i) for i in range(phi.ndim))


def reinitialize(
    phi: np.ndarray, dx: float, dt: float = 0.1, n_steps: int = 5
) -> np.ndarray:
    """Level Set関数を符号付き距離関数に再初期化

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        dt: 仮想時間の時間刻み幅
        n_steps: 再初期化の反復回数

    Returns:
        再初期化されたLevel Set関数
    """
    result = phi.copy()

    for _ in range(n_steps):
        # 勾配を計算
        grad = np.array(np.gradient(result, dx))
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))

        # 時間発展方程式を解く
        result = result - dt * np.sign(phi) * (grad_norm - 1.0)

        # 数値的な安定化のために少しぼかす
        result = gaussian_filter(result, sigma=0.5 * dx)

    return result


def extend_velocity(
    velocity: np.ndarray, phi: np.ndarray, dx: float, n_steps: int = 5
) -> np.ndarray:
    """界面の法線方向に速度場を拡張

    Args:
        velocity: 拡張する速度場
        phi: Level Set関数
        dx: グリッド間隔
        n_steps: 拡張の反復回数

    Returns:
        拡張された速度場
    """
    result = velocity.copy()

    for _ in range(n_steps):
        # Level Set関数の勾配（法線方向）を計算
        grad = np.array(np.gradient(phi, dx))
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))
        grad_norm = np.maximum(grad_norm, 1e-10)
        normal = grad / grad_norm

        # 速度場の勾配を計算
        vel_grad = np.array(np.gradient(result, dx))

        # 法線方向の速度勾配をゼロにする
        result = result - dt * sum(normal[i] * vel_grad[i] for i in range(phi.ndim))

        # 数値的な安定化
        result = gaussian_filter(result, sigma=0.5 * dx)

    return result


def compute_volume(phi: np.ndarray, dx: float, epsilon: float = 1.0e-2) -> float:
    """Level Set関数から体積を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        epsilon: Heaviside関数の正則化パラメータ

    Returns:
        計算された体積
    """
    return np.sum(heaviside(phi, epsilon)) * dx**phi.ndim


def compute_area(phi: np.ndarray, dx: float, epsilon: float = 1.0e-2) -> float:
    """Level Set関数から界面の面積を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        epsilon: Delta関数の正則化パラメータ

    Returns:
        計算された界面の面積
    """
    return np.sum(delta(phi, epsilon)) * dx**phi.ndim
