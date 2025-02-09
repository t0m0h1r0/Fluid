"""Level Set場の再初期化を提供するモジュール

このモジュールは、Level Set関数を符号付き距離関数として再初期化する機能を提供します。
Hamilton-Jacobiタイプの方程式を解くことで、界面の位置を保ったまま
符号付き距離関数の性質を回復します。
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

from .field import LevelSetField


def reinitialize_levelset(
    levelset: LevelSetField,
    dt: float = 0.1,
    n_steps: int = 5,
    width: float = 2.0,
    smooth_width: float = 0.5,
) -> LevelSetField:
    """Level Set関数を再初期化

    Args:
        levelset: 再初期化するLevel Set場
        dt: 仮想時間の時間刻み幅
        n_steps: 反復回数
        width: 再初期化を行う界面近傍の幅
        smooth_width: スムージングの幅（グリッド間隔に対する比）

    Returns:
        再初期化されたLevel Set場
    """
    # 結果を格納する配列
    result = levelset.copy()

    # 初期の符号を保存
    sign = np.sign(result.data)
    sign = smooth_sign(sign, levelset.dx * smooth_width)

    # バンド幅の設定（グリッド間隔に対する比）
    band_width = width * levelset.dx

    # 仮想時間発展による再初期化
    for _ in range(n_steps):
        # 現在の勾配を計算
        gradients = compute_gradients(result.data, levelset.dx)
        grad_norm = compute_gradient_norm(gradients)

        # 更新量の計算
        update = sign * (1.0 - grad_norm)

        # 界面近傍でのみ更新
        mask = np.abs(result.data) <= band_width
        result.data[mask] += dt * update[mask]

        # 数値的な安定化のためのスムージング
        result.data = gaussian_filter(
            result.data, sigma=smooth_width * levelset.dx / levelset.dx
        )

    return result


def compute_gradients(phi: np.ndarray, dx: float) -> Tuple[np.ndarray, ...]:
    """勾配を計算（Godunov upwindスキーム）

    Args:
        phi: Level Set関数
        dx: グリッド間隔

    Returns:
        各方向の勾配のタプル
    """
    gradients = []
    for axis in range(phi.ndim):
        # 前方差分
        phi_forward = np.roll(phi, -1, axis=axis)
        D_plus = (phi_forward - phi) / dx

        # 後方差分
        phi_backward = np.roll(phi, 1, axis=axis)
        D_minus = (phi - phi_backward) / dx

        # Godunovスキームのための勾配選択
        grad_pos = np.maximum(np.maximum(-D_minus, 0) ** 2, np.minimum(D_plus, 0) ** 2)
        grad_neg = np.maximum(np.minimum(-D_minus, 0) ** 2, np.maximum(D_plus, 0) ** 2)

        # 符号に応じて適切な勾配を選択
        gradient = np.where(phi >= 0, grad_pos, grad_neg)
        gradients.append(np.sqrt(gradient))

    return tuple(gradients)


def compute_gradient_norm(
    gradients: Tuple[np.ndarray, ...], epsilon: float = 1e-10
) -> np.ndarray:
    """勾配ベクトルのノルムを計算

    Args:
        gradients: 各方向の勾配のタプル
        epsilon: ゼロ除算を防ぐための小さな値

    Returns:
        勾配ベクトルのノルム（最小値はepsilon）
    """
    # 各方向の勾配の二乗和を計算
    sum_squares = sum(g * g for g in gradients)

    # 数値安定性のための最小値制限
    return np.sqrt(np.maximum(sum_squares, epsilon * epsilon))


def smooth_sign(sign: np.ndarray, epsilon: float) -> np.ndarray:
    """符号関数をスムージング

    Args:
        sign: 符号の配列
        epsilon: スムージングの幅

    Returns:
        スムージングされた符号
    """
    return sign / np.sqrt(sign * sign + epsilon * epsilon)
