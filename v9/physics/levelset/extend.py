"""Level Set法の拡張速度場を提供するモジュール

このモジュールは、界面の法線方向に物理量を拡張する機能を提供します。
界面から離れた領域でも適切な速度場を定義するために使用されます。
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

from core.field import VectorField, ScalarField
from .field import LevelSetField
from .reinitialize import compute_gradient_norm


def extend_velocity(
    velocity: VectorField,
    levelset: LevelSetField,
    width: float = 3.0,  # バンド幅（グリッド間隔に対する比）
    dt: float = 0.1,  # 仮想時間の時間刻み幅
    n_steps: int = 5,  # 反復回数
    smooth_width: float = 0.5,  # スムージングの幅
) -> VectorField:
    """速度場を界面の法線方向に拡張

    Args:
        velocity: 拡張する速度場
        levelset: Level Set場
        width: 拡張を行う界面近傍の幅
        dt: 仮想時間の時間刻み幅
        n_steps: 反復回数
        smooth_width: スムージングの幅

    Returns:
        拡張された速度場
    """
    # 結果を格納する速度場
    result = velocity.copy()

    # Level Set関数の勾配（法線方向）を計算
    gradients = compute_levelset_gradients(levelset)
    grad_norm = compute_gradient_norm(gradients)
    grad_norm = np.maximum(grad_norm, 1e-10)  # ゼロ除算を防ぐ

    # 正規化された法線ベクトル
    normals = tuple(g / grad_norm for g in gradients)

    # バンド幅の設定
    band_width = width * levelset.dx

    # 各速度成分について拡張
    for i, component in enumerate(result.components):
        # 界面近傍でのみ更新
        mask = np.abs(levelset.data) <= band_width

        # 仮想時間発展による拡張
        for _ in range(n_steps):
            # 速度場の勾配を計算
            vel_grads = compute_velocity_gradients(component.data, levelset.dx)

            # 法線方向の勾配をゼロにする更新
            update = sum(n * g for n, g in zip(normals, vel_grads))
            component.data[mask] -= dt * update[mask]

            # 数値的な安定化のためのスムージング
            component.data = gaussian_filter(component.data, sigma=smooth_width)

    return result


def compute_levelset_gradients(levelset: LevelSetField) -> Tuple[np.ndarray, ...]:
    """Level Set関数の勾配を計算

    Args:
        levelset: Level Set場

    Returns:
        各方向の勾配のタプル
    """
    gradients = []
    for axis in range(levelset.ndim):
        grad = np.gradient(levelset.data, levelset.dx, axis=axis)
        gradients.append(grad)
    return tuple(gradients)


def compute_velocity_gradients(
    velocity: np.ndarray, dx: float
) -> Tuple[np.ndarray, ...]:
    """速度場の勾配を計算

    Args:
        velocity: 速度場の1成分
        dx: グリッド間隔

    Returns:
        各方向の勾配のタプル
    """
    gradients = []
    for axis in range(velocity.ndim):
        # 中心差分による勾配計算
        grad = np.gradient(velocity, dx, axis=axis)
        gradients.append(grad)
    return tuple(gradients)


def extend_scalar(
    scalar: ScalarField,
    levelset: LevelSetField,
    width: float = 3.0,
    dt: float = 0.1,
    n_steps: int = 5,
    smooth_width: float = 0.5,
) -> ScalarField:
    """スカラー場を界面の法線方向に拡張

    Args:
        scalar: 拡張するスカラー場
        levelset: Level Set場
        width: 拡張を行う界面近傍の幅
        dt: 仮想時間の時間刻み幅
        n_steps: 反復回数
        smooth_width: スムージングの幅

    Returns:
        拡張されたスカラー場
    """
    # 結果を格納するスカラー場
    result = scalar.copy()

    # Level Set関数の勾配（法線方向）を計算
    gradients = compute_levelset_gradients(levelset)
    grad_norm = compute_gradient_norm(gradients)
    grad_norm = np.maximum(grad_norm, 1e-10)

    # 正規化された法線ベクトル
    normals = tuple(g / grad_norm for g in gradients)

    # バンド幅の設定
    band_width = width * levelset.dx

    # 界面近傍でのみ更新
    mask = np.abs(levelset.data) <= band_width

    # 仮想時間発展による拡張
    for _ in range(n_steps):
        # スカラー場の勾配を計算
        scalar_grads = compute_velocity_gradients(result.data, levelset.dx)

        # 法線方向の勾配をゼロにする更新
        update = sum(n * g for n, g in zip(normals, scalar_grads))
        result.data[mask] -= dt * update[mask]

        # 数値的な安定化のためのスムージング
        result.data = gaussian_filter(result.data, sigma=smooth_width)

    return result
