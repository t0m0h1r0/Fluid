"""Level Set関数の再初期化を提供するモジュール

このモジュールは、Level Set関数を符号付き距離関数として再初期化する
ための高精度な数値計算手法を実装します。
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import List, Optional

from .field import LevelSetField
from .utils import delta


def reinitialize_levelset(
    levelset: LevelSetField,
    dt: Optional[float] = None,
    n_steps: Optional[int] = None,
    method: str = "fast_marching",
) -> LevelSetField:
    """Level Set関数を再初期化

    Args:
        levelset: 再初期化するLevel Set場
        dt: 仮想時間の時間刻み幅（Noneの場合は設定から取得）
        n_steps: 反復回数（Noneの場合は設定から取得）
        method: 再初期化手法

    Returns:
        再初期化されたLevel Set場
    """
    # パラメータの決定
    dt = dt or levelset.params.reinit_dt
    n_steps = n_steps or levelset.params.reinit_steps

    # 初期の符号を保存
    result = levelset.copy()
    original_sign = np.sign(result.data)

    # 再初期化手法の選択
    if method == "fast_marching":
        result.data = _fast_marching_reinit(
            result.data,
            result.dx,
            dt=dt,
            n_steps=n_steps,
            epsilon=levelset.params.epsilon,
        )
    elif method == "pde":
        result.data = _pde_reinit(
            result.data,
            result.dx,
            dt=dt,
            n_steps=n_steps,
            epsilon=levelset.params.epsilon,
        )
    else:
        raise ValueError(f"未対応の再初期化手法: {method}")

    # 初期の符号を維持
    result.data = np.copysign(result.data, original_sign)

    return result


def _pde_reinit(
    phi: np.ndarray,
    dx: float,
    dt: float = 0.1,
    n_steps: int = 5,
    epsilon: float = 1.0e-2,
) -> np.ndarray:
    """偏微分方程式に基づく再初期化

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        dt: 仮想時間の時間刻み幅
        n_steps: 再初期化の反復回数
        epsilon: 正則化パラメータ

    Returns:
        再初期化されたLevel Set関数
    """
    result = phi.copy()
    sign = np.sign(result)

    for _ in range(n_steps):
        # 勾配を計算
        grad = np.array(np.gradient(result, dx))
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))

        # 時間発展方程式を解く
        result = result - dt * sign * (grad_norm - 1.0)

        # 数値的な安定化のためにぼかす
        result = gaussian_filter(result, sigma=0.5 * dx)

    return result


def _fast_marching_reinit(
    phi: np.ndarray,
    dx: float,
    dt: float = 0.1,
    n_steps: int = 5,
    epsilon: float = 1.0e-2,
) -> np.ndarray:
    """高速行進法に基づく再初期化

    Args:
        phi: Level Set関数
        dx: グリッド間隔
        dt: 仮想時間の時間刻み幅
        n_steps: 再初期化の反復回数
        epsilon: 正則化パラメータ

    Returns:
        再初期化されたLevel Set関数
    """
    result = phi.copy()
    sign = np.sign(result)

    # アクティブな点のマスクを作成
    delta_vals = delta(result, epsilon)
    active_mask = delta_vals > 0

    # 各点の符号付き距離を計算
    for _ in range(n_steps):
        # アクティブな点周辺の勾配を計算
        active_points = np.argwhere(active_mask)
        for point in active_points:
            # 最近傍点からの距離を計算
            distances = _compute_signed_distance(point, result, sign, dx)
            # 最小の距離を選択
            result[tuple(point)] = sign[tuple(point)] * np.min(np.abs(distances))

        # 数値的な安定化のためにぼかす
        result = gaussian_filter(result, sigma=0.5 * dx)

    return result


def _compute_signed_distance(
    point: np.ndarray, phi: np.ndarray, sign: np.ndarray, dx: float
) -> List[float]:
    """指定された点の符号付き距離を計算

    Args:
        point: 距離を計算する点の座標
        phi: Level Set関数
        sign: 符号
        dx: グリッド間隔

    Returns:
        近傍点からの符号付き距離のリスト
    """
    distances = []
    ndim = phi.ndim

    # 各近傍点について
    for offsets in np.ndindex(tuple([3] * ndim)):
        # オフセット補正（中心点を除外）
        if all(o == 1 for o in offsets):
            continue

        # 近傍点の座標を計算
        neighbor = point + np.array(offsets) - 1

        # 境界チェック
        if all(0 <= n < phi.shape[i] for i, n in enumerate(neighbor)):
            # 近傍点からの距離を計算
            distance = np.linalg.norm((neighbor - point) * dx)
            signed_dist = sign[tuple(point)] * distance
            distances.append(signed_dist)

    return distances if distances else [0.0]
