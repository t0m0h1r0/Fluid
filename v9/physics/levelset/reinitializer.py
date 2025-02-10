import numpy as np
from scipy.ndimage import gaussian_filter
from .field import LevelSetField


def reinitialize_levelset(
    levelset: LevelSetField,
    dt: float = 0.1,
    n_steps: int = 5,
    method: str = "fast_marching",
) -> LevelSetField:
    """Level Set関数を再初期化"""
    if method == "fast_marching":
        return _fast_marching_reinit(levelset, dt, n_steps)
    elif method == "pde":
        return _pde_reinit(levelset, dt, n_steps)
    else:
        raise ValueError(f"未対応の再初期化手法: {method}")


def _fast_marching_reinit(
    levelset: LevelSetField, dt: float, n_steps: int
) -> LevelSetField:
    """高速行進法による再初期化"""
    result = levelset.data.copy()
    sign = np.sign(result)

    for _ in range(n_steps):
        # 界面近傍の点を特定
        interface_points = np.abs(result) < levelset.params.epsilon

        for point in np.argwhere(interface_points):
            # 最近傍点の符号付き距離を計算
            distances = _compute_signed_distance(point, result, sign, levelset.dx)
            result[tuple(point)] = sign[tuple(point)] * np.min(np.abs(distances))

        # 数値的安定化のためにガウシアンフィルタを適用
        result = gaussian_filter(result, sigma=0.5 * levelset.dx)

    return LevelSetField(data=result, dx=levelset.dx, params=levelset.params)


def _pde_reinit(levelset: LevelSetField, dt: float, n_steps: int) -> LevelSetField:
    """PDE法による再初期化"""
    result = levelset.data.copy()
    sign = np.sign(result)

    for _ in range(n_steps):
        # 勾配を計算
        grad = np.array(np.gradient(result, levelset.dx))
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))

        # 時間発展
        result = result - dt * sign * (grad_norm - 1.0)

        # 数値的安定化
        result = gaussian_filter(result, sigma=0.5 * levelset.dx)

    return LevelSetField(data=result, dx=levelset.dx, params=levelset.params)


def _compute_signed_distance(
    point: np.ndarray, phi: np.ndarray, sign: np.ndarray, dx: float
) -> np.ndarray:
    """指定された点の符号付き距離を計算"""
    # 近傍点からの距離を計算する実装（省略）
    pass
