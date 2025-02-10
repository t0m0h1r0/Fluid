"""Level Set法のユーティリティ関数を提供するモジュール

このモジュールは、Level Set法で必要となる基本的な数値計算関数と
物性値管理の機能を提供します。
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, Literal


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


def interpolate_property(
    phi: np.ndarray,
    prop1: float,
    prop2: float,
    epsilon: float = 1.0e-2,
    method: Literal["arithmetic", "harmonic"] = "arithmetic",
) -> np.ndarray:
    """物性値を補間

    Args:
        phi: Level Set関数
        prop1: 第1相の物性値
        prop2: 第2相の物性値
        epsilon: 正則化パラメータ
        method: 補間方法

    Returns:
        補間された物性値場
    """
    H = heaviside(phi, epsilon)

    if method == "arithmetic":
        # 算術平均
        return prop1 * H + prop2 * (1 - H)
    else:
        # 調和平均
        return 1.0 / (H / prop1 + (1 - H) / prop2)


def compute_property_jump(
    prop1: float, prop2: float, method: Literal["absolute", "relative"] = "absolute"
) -> float:
    """物性値のジャンプを計算

    Args:
        prop1: 第1相の物性値
        prop2: 第2相の物性値
        method: ジャンプ計算方法

    Returns:
        計算されたジャンプ値
    """
    if method == "absolute":
        return abs(prop1 - prop2)
    else:
        # 相対ジャンプ
        return abs(prop1 - prop2) / max(abs(prop1), abs(prop2))


def compute_interface_statistics(
    data: np.ndarray, phi: np.ndarray, epsilon: float = 1.0e-2
) -> Dict[str, float]:
    """界面近傍の統計情報を計算

    Args:
        data: データ
        phi: Level Set関数
        epsilon: 正則化パラメータ

    Returns:
        統計情報の辞書
    """
    delta_vals = delta(phi, epsilon)
    interface_mask = delta_vals > 0

    if not np.any(interface_mask):
        return {"count": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}

    interface_data = data[interface_mask]

    return {
        "count": len(interface_data),
        "min": float(np.min(interface_data)),
        "max": float(np.max(interface_data)),
        "mean": float(np.mean(interface_data)),
        "std": float(np.std(interface_data)),
    }


def compute_curvature(
    phi: np.ndarray, dx: float, epsilon: float = 1.0e-2
) -> np.ndarray:
    """界面の曲率を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        epsilon: 正則化パラメータ

    Returns:
        界面の曲率
    """
    # 各方向の勾配を計算
    grad = np.array(np.gradient(phi, dx))
    grad_norm = np.sqrt(np.sum(grad**2, axis=0))
    grad_norm = np.maximum(grad_norm, epsilon)  # ゼロ除算を防ぐ

    # 正規化された勾配
    grad_normalized = grad / grad_norm

    # 発散を計算して曲率を得る
    return sum(np.gradient(grad_normalized[i], dx, axis=i) for i in range(phi.ndim))


def compute_volume(phi: np.ndarray, dx: float, epsilon: float = 1.0e-2) -> float:
    """Level Set関数から体積を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        epsilon: 正則化パラメータ

    Returns:
        計算された体積
    """
    return np.sum(heaviside(phi, epsilon)) * dx**phi.ndim


def compute_area(phi: np.ndarray, dx: float, epsilon: float = 1.0e-2) -> float:
    """Level Set関数から界面の面積を計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        epsilon: 正則化パラメータ

    Returns:
        計算された界面の面積
    """
    return np.sum(delta(phi, epsilon)) * dx**phi.ndim


def compute_interface_gradient(
    phi: np.ndarray, dx: float, epsilon: float = 1.0e-2
) -> np.ndarray:
    """界面の法線ベクトルを計算

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        epsilon: 正則化パラメータ

    Returns:
        界面の法線ベクトル
    """
    # 勾配を計算
    grad = np.array(np.gradient(phi, dx))
    grad_norm = np.sqrt(np.sum(grad**2, axis=0))
    grad_norm = np.maximum(grad_norm, epsilon)  # ゼロ除算を防ぐ

    # 正規化された勾配（法線ベクトル）
    return grad / grad_norm


def extend_velocity(
    velocity: np.ndarray, phi: np.ndarray, dx: float, dt: float = 0.1, n_steps: int = 5
) -> np.ndarray:
    """界面の法線方向に速度場を拡張

    Args:
        velocity: 拡張する速度場
        phi: Level Set関数
        dx: グリッド間隔
        dt: 仮想時間の時間刻み幅
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

        # 数値的な安定化のためにぼかす
        result = gaussian_filter(result, sigma=0.5 * dx)

    return result


def validate_signed_distance_function(
    phi: np.ndarray, dx: float, tolerance: float = 1e-2
) -> bool:
    """符号付き距離関数としての性質を検証

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        tolerance: 許容誤差

    Returns:
        符号付き距離関数の条件を満たすかどうか
    """
    # 勾配の大きさが1に近いかチェック
    grad = np.array(np.gradient(phi, dx))
    grad_norm = np.sqrt(np.sum(grad**2, axis=0))

    # 勾配の大きさが1にどれだけ近いか
    is_unit_gradient = np.abs(grad_norm - 1.0)

    # 界面の幅をチェック
    delta_values = delta(phi)
    interface_width = np.sum(delta_values > 0)

    # 両条件を確認
    return np.mean(is_unit_gradient) < tolerance and interface_width < tolerance
