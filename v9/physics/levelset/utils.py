"""Level Set法のユーティリティ関数を提供するモジュール"""

import numpy as np


def compute_density(
    phi: np.ndarray, rho1: float, rho2: float, epsilon: float = 1.0e-2
) -> np.ndarray:
    """密度場を計算

    Args:
        phi: Level Set関数
        rho1: 第1相の密度
        rho2: 第2相の密度
        epsilon: 界面の厚さ

    Returns:
        密度場
    """
    H = heaviside(phi, epsilon)
    return rho1 * H + rho2 * (1 - H)


def compute_viscosity(
    phi: np.ndarray,
    mu1: float,
    mu2: float,
    epsilon: float = 1.0e-2,
    use_harmonic: bool = True,
) -> np.ndarray:
    """粘性係数場を計算

    Args:
        phi: Level Set関数
        mu1: 第1相の粘性係数
        mu2: 第2相の粘性係数
        epsilon: 界面の厚さ
        use_harmonic: 調和平均を使用するかどうか

    Returns:
        粘性係数場
    """
    H = heaviside(phi, epsilon)
    if use_harmonic:
        # 調和平均（界面での応力の連続性に適している）
        return 1.0 / (H / mu1 + (1 - H) / mu2)
    else:
        # 算術平均
        return mu1 * H + mu2 * (1 - H)


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
        # 法線方向を計算
        grad = np.array(np.gradient(phi, dx))
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))
        grad_norm = np.maximum(grad_norm, 1e-10)
        normal = grad / grad_norm

        # 速度勾配を計算
        vel_grad = np.array(np.gradient(result, dx))

        # 法線方向の速度勾配を解消
        result -= dt * sum(normal[i] * vel_grad[i] for i in range(phi.ndim))

    return result
