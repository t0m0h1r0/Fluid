"""WENOスキームの滑らかさ指標を計算するモジュール

このモジュールは、WENOスキームで使用される滑らかさ指標（smoothness indicators）の
計算を担当します。
"""

from typing import List, Optional
import numpy as np
import numpy.typing as npt


class SmoothnessIndicator:
    """WENOスキームの滑らかさ指標を計算するクラス"""

    def __init__(self, dx: float = 1.0):
        """滑らかさ指標計算器を初期化

        Args:
            dx: グリッド間隔
        """
        self._dx = dx
        self._cache: Optional[List[npt.NDArray[np.float64]]] = None

    def compute(
        self,
        data: npt.NDArray[np.float64],
        coeffs: List[np.ndarray],
        axis: int = -1,
    ) -> List[npt.NDArray[np.float64]]:
        """滑らかさ指標を計算

        Args:
            data: 入力データ配列
            coeffs: 滑らかさ指標の計算係数
            axis: 計算を行う軸

        Returns:
            各ステンシルの滑らかさ指標のリスト
        """
        # 軸の正規化
        axis = axis if axis >= 0 else data.ndim + axis

        # 結果を格納するリスト
        beta = []

        # 各ステンシルについて滑らかさ指標を計算
        for stencil_coeffs in coeffs:
            # このステンシルの滑らかさ指標を初期化
            beta_r = np.zeros_like(data, dtype=np.float64)

            # 各微分の次数について
            for k, deriv_coeffs in enumerate(stencil_coeffs):
                # 差分近似の計算
                diff = np.zeros_like(data)
                for j, coeff in enumerate(deriv_coeffs):
                    diff += coeff * np.roll(data, j - 1, axis=axis)

                # スケーリング係数（dx^(2k-1)）
                scale = self._dx ** (2 * k - 1)

                # 滑らかさ指標に寄与を加算
                beta_r += scale * diff * diff

            beta.append(beta_r)

        # キャッシュの更新
        self._cache = beta

        return beta

    def compute_normalized(
        self,
        data: npt.NDArray[np.float64],
        coeffs: List[np.ndarray],
        axis: int = -1,
        p: float = 2.0,
    ) -> List[npt.NDArray[np.float64]]:
        """正規化された滑らかさ指標を計算

        Args:
            data: 入力データ配列
            coeffs: 滑らかさ指標の計算係数
            axis: 計算を行う軸
            p: 正規化のための指数（デフォルト: 2.0）

        Returns:
            正規化された滑らかさ指標のリスト
        """
        # 滑らかさ指標の計算
        beta = self.compute(data, coeffs, axis)

        # 最大値による正規化
        beta_max = max(np.max(b) for b in beta)
        if beta_max > 0:
            beta = [b / beta_max for b in beta]

        # べき乗による強調
        if p != 1.0:
            beta = [b**p for b in beta]

        return beta

    def get_last_computed(self) -> Optional[List[npt.NDArray[np.float64]]]:
        """最後に計算された滑らかさ指標を取得"""
        return self._cache

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache = None
