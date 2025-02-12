"""Level Set法での入力検証を提供するモジュール

このモジュールは、Level Set法で使用される各種入力の
検証機能を提供します。
"""

from typing import Tuple
import numpy as np


def validate_shape(shape: Tuple[int, ...]) -> None:
    """グリッド形状を検証

    Args:
        shape: 検証するグリッド形状

    Raises:
        ValueError: 無効な形状が指定された場合
        TypeError: 無効な型が指定された場合
    """
    if not isinstance(shape, tuple):
        raise TypeError("shapeはtupleである必要があります")

    if not shape:
        raise ValueError("shapeは空であってはいけません")

    if not all(isinstance(s, int) for s in shape):
        raise TypeError("shapeの要素は整数である必要があります")

    if not all(s > 0 for s in shape):
        raise ValueError("shapeの要素は正の値である必要があります")

    if len(shape) not in [2, 3]:
        raise ValueError("2次元または3次元の形状である必要があります")


def validate_dx(dx: float) -> None:
    """グリッド間隔を検証

    Args:
        dx: 検証するグリッド間隔

    Raises:
        ValueError: 無効な値が指定された場合
        TypeError: 無効な型が指定された場合
    """
    if not isinstance(dx, (int, float)):
        raise TypeError("dxは数値である必要があります")

    if dx <= 0:
        raise ValueError("dxは正の値である必要があります")


def validate_array(arr: np.ndarray, name: str = "array") -> None:
    """NumPy配列を検証

    Args:
        arr: 検証する配列
        name: エラーメッセージで使用する配列の名前

    Raises:
        ValueError: 無効な配列が指定された場合
        TypeError: 無効な型が指定された場合
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name}はnumpy配列である必要があります")

    if arr.ndim not in [2, 3]:
        raise ValueError(f"{name}は2次元または3次元である必要があります")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name}に無限大または非数が含まれています")


def validate_epsilon(epsilon: float) -> None:
    """界面の厚さパラメータを検証

    Args:
        epsilon: 検証するパラメータ

    Raises:
        ValueError: 無効な値が指定された場合
        TypeError: 無効な型が指定された場合
    """
    if not isinstance(epsilon, (int, float)):
        raise TypeError("epsilonは数値である必要があります")

    if epsilon <= 0:
        raise ValueError("epsilonは正の値である必要があります")


def validate_method(method: str, valid_methods: Tuple[str, ...]) -> None:
    """計算手法を検証

    Args:
        method: 検証する手法名
        valid_methods: 有効な手法名のタプル

    Raises:
        ValueError: 無効な手法が指定された場合
    """
    if method not in valid_methods:
        methods_str = "', '".join(valid_methods)
        raise ValueError(f"無効な計算手法です。有効な値: '{methods_str}'")
