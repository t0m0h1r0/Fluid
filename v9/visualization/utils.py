"""可視化のユーティリティ関数を提供するモジュール

共通の処理や補助関数を定義します。
"""

import numpy as np
from typing import Tuple, List
from core.field import ScalarField
from core.field.vector import VectorField


def prepare_2d_slice(
    data: np.ndarray | ScalarField | VectorField,
    slice_axis: int = 2,
    slice_index: int = None,
) -> np.ndarray:
    """3Dデータから2Dスライスを取得

    Args:
        data: 入力データ
        slice_axis: スライスする軸
        slice_index: スライスのインデックス（デフォルトは中央）

    Returns:
        2Dスライス
    """
    # データの型に応じて処理を変更
    if isinstance(data, (ScalarField, VectorField)):
        data = data.data

    # 2Dデータの場合はそのまま返す
    if data.ndim == 2:
        return data

    # スライスのインデックスを決定
    if slice_index is None:
        slice_index = data.shape[slice_axis] // 2

    # スライスを取得
    slices = [slice(None)] * data.ndim
    slices[slice_axis] = slice_index

    return data[tuple(slices)]


def compute_data_range(
    data: np.ndarray, symmetric: bool = False
) -> Tuple[float, float]:
    """データの範囲を計算

    Args:
        data: 入力データ
        symmetric: 対称な範囲にするかどうか

    Returns:
        (最小値, 最大値)のタプル
    """
    data_min = np.min(data)
    data_max = np.max(data)

    if symmetric:
        abs_max = max(abs(data_min), abs(data_max))
        return -abs_max, abs_max

    return data_min, data_max


def compute_vector_magnitude(vector_components: List[np.ndarray]) -> np.ndarray:
    """ベクトル場の大きさを計算

    Args:
        vector_components: ベクトルの各成分

    Returns:
        ベクトルの大きさ
    """
    if len(vector_components) < 2:
        raise ValueError("少なくとも2つの成分が必要です")

    return np.sqrt(sum(c**2 for c in vector_components[:2]))


def create_grid(data_shape: Tuple[int, int]):
    """グリッドを生成

    Args:
        data_shape: データの形状

    Returns:
        x, y座標のメッシュグリッド
    """
    nx, ny = data_shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return x, y
