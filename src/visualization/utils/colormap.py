"""カラーマップ操作のユーティリティを提供するモジュール

このモジュールは、データの可視化に使用するカラーマップの
生成と操作に関する機能を提供します。
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize


def create_custom_colormap(
    colors: List[str], name: str = "custom", n_bins: int = 256
) -> LinearSegmentedColormap:
    """カスタムカラーマップを作成

    Args:
        colors: カラーのリスト
        name: カラーマップの名前
        n_bins: 色の分割数

    Returns:
        作成されたカラーマップ
    """
    return LinearSegmentedColormap.from_list(name, colors, N=n_bins)


def create_diverging_colormap(
    neutral_point: float = 0.0,
    vmin: float = -1.0,
    vmax: float = 1.0,
    colors: Optional[List[str]] = None,
) -> LinearSegmentedColormap:
    """発散型カラーマップを作成

    Args:
        neutral_point: 中立点の値
        vmin: 最小値
        vmax: 最大値
        colors: カラーのリスト（省略時は blue-white-red）

    Returns:
        作成されたカラーマップ
    """
    if colors is None:
        colors = ["blue", "white", "red"]

    # 中立点の相対位置を計算
    total_range = vmax - vmin
    if total_range <= 0:
        raise ValueError("無効なデータ範囲です")

    neutral_pos = (neutral_point - vmin) / total_range
    neutral_pos = np.clip(neutral_pos, 0, 1)

    # カラーマップの作成
    if neutral_pos == 0.5:
        # 対称なカラーマップ
        return create_custom_colormap(colors)
    else:
        # 非対称なカラーマップ
        positions = [0, neutral_pos, 1]
        return LinearSegmentedColormap.from_list(
            "diverging", list(zip(positions, colors))
        )


def apply_colormap(
    data: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    symmetric: bool = False,
    alpha: Optional[float] = None,
) -> np.ndarray:
    """データにカラーマップを適用

    Args:
        data: 入力データ
        vmin: 最小値（省略時は自動計算）
        vmax: 最大値（省略時は自動計算）
        cmap: カラーマップ名
        symmetric: 対称な範囲にするかどうか
        alpha: 透明度

    Returns:
        RGBA形式のカラー配列
    """
    # データ範囲の計算
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    if symmetric:
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    # 正規化とカラーマップの適用
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapper = plt.get_cmap(cmap)
    colors = mapper(norm(data))

    # 透明度の設定
    if alpha is not None:
        colors[..., 3] = alpha

    return colors
