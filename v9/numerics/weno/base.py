"""WENOスキームの基底クラスと抽象インターフェースを提供

このモジュールは、Weighted Essentially Non-Oscillatory (WENO) スキームの
基本的なインターフェースと共通機能を定義します。

References:
    [1] Shu, Chi-Wang. "High order weighted essentially nonoscillatory
        schemes for convection dominated problems."
        SIAM review 51.1 (2009): 82-126.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence
import numpy as np
import numpy.typing as npt


class WENOBase(ABC):
    """WENOスキームの基底クラス

    このクラスは、WENOスキームの基本的なインターフェースを定義し、
    共通の機能を提供します。
    """

    def __init__(self, order: int = 5, epsilon: float = 1e-6):
        """WENOスキームを初期化

        Args:
            order: スキームの次数（デフォルト: 5）
            epsilon: ゼロ除算を防ぐための小さな値
        """
        self._order = order
        self._epsilon = epsilon
        self._stencil_size = (order + 1) // 2
        self._cache: Dict[str, Any] = {}

    @property
    def order(self) -> int:
        """スキームの次数を取得"""
        return self._order

    @property
    def stencil_size(self) -> int:
        """ステンシルサイズを取得"""
        return self._stencil_size

    @abstractmethod
    def reconstruct(
        self, data: npt.NDArray[np.float64], axis: int = -1
    ) -> npt.NDArray[np.float64]:
        """WENOスキームによる再構成を実行

        Args:
            data: 入力データ配列
            axis: 再構成を行う軸

        Returns:
            再構成された値の配列
        """
        pass

    @abstractmethod
    def compute_smoothness_indicators(
        self, data: npt.NDArray[np.float64], axis: int = -1
    ) -> Sequence[npt.NDArray[np.float64]]:
        """滑らかさ指標を計算

        Args:
            data: 入力データ配列
            axis: 計算を行う軸

        Returns:
            各ステンシルの滑らかさ指標
        """
        pass

    def _validate_input(self, data: npt.NDArray[np.float64], axis: int) -> None:
        """入力データの妥当性を検証

        Args:
            data: 入力データ配列
            axis: 処理する軸

        Raises:
            ValueError: 無効な入力が指定された場合
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("入力はNumPy配列である必要があります")

        if data.ndim < 1:
            raise ValueError("入力は少なくとも1次元である必要があります")

        if not -data.ndim <= axis < data.ndim:
            raise ValueError(f"無効な軸です: {axis}")

        if data.shape[axis] < self._stencil_size:
            raise ValueError(
                f"軸{axis}のサイズ({data.shape[axis]})が"
                f"ステンシルサイズ({self._stencil_size})より小さいです"
            )

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()

    def get_status(self) -> Dict[str, Any]:
        """WENOスキームの状態を取得"""
        return {
            "order": self._order,
            "stencil_size": self._stencil_size,
            "epsilon": self._epsilon,
            "cache_size": len(self._cache),
        }
