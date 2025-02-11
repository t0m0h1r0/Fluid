"""WENOスキームの重み係数を計算するモジュール

このモジュールは、WENOスキームの非線形重み係数の計算を担当します。
これらの重み係数は、滑らかさ指標に基づいて各ステンシルの寄与を決定します。
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import numpy.typing as npt


class WeightCalculator:
    """WENOスキームの重み係数を計算するクラス"""

    def __init__(self, epsilon: float = 1e-6, p: float = 2.0):
        """重み係数計算器を初期化

        Args:
            epsilon: ゼロ除算を防ぐための小さな値
            p: 非線形重みの指数
        """
        self._epsilon = epsilon
        self._p = p
        self._cache: Dict[str, Any] = {}

    def compute_weights(
        self,
        beta: List[npt.NDArray[np.float64]],
        optimal_weights: npt.NDArray[np.float64],
    ) -> Tuple[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """非線形重み係数を計算

        Args:
            beta: 各ステンシルの滑らかさ指標
            optimal_weights: 理想重み係数

        Returns:
            (非線形重み係数のリスト, 正規化係数）のタプル
        """
        # アルファの計算 (α_k = d_k / (ε + β_k)^p)
        alpha = []
        for b, d in zip(beta, optimal_weights):
            alpha_k = d / (self._epsilon + b) ** self._p
            alpha.append(alpha_k)

        # 正規化係数の計算
        omega_sum = sum(alpha)

        # 正規化された重み係数の計算
        omega = [a / omega_sum for a in alpha]

        # 結果をキャッシュ
        self._cache = {"alpha": alpha, "omega": omega, "omega_sum": omega_sum}

        return omega, omega_sum

    def compute_mapped_weights(
        self,
        beta: List[npt.NDArray[np.float64]],
        optimal_weights: npt.NDArray[np.float64],
        mapping_function: Optional[str] = None,
    ) -> Tuple[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """マッピング関数を使用して非線形重み係数を計算

        Args:
            beta: 各ステンシルの滑らかさ指標
            optimal_weights: 理想重み係数
            mapping_function: マッピング関数の種類（'henrick'または'borges'）

        Returns:
            (非線形重み係数のリスト, 正規化係数）のタプル
        """
        # まず通常の重み係数を計算
        omega, omega_sum = self.compute_weights(beta, optimal_weights)

        if mapping_function is None:
            return omega, omega_sum

        # マッピング関数の適用
        if mapping_function == "henrick":
            omega = self._henrick_mapping(omega, optimal_weights)
        elif mapping_function == "borges":
            omega = self._borges_mapping(omega, optimal_weights)
        else:
            raise ValueError(f"未対応のマッピング関数です: {mapping_function}")

        # 正規化係数の再計算
        omega_sum = sum(o for o in omega)

        return omega, omega_sum

    def _henrick_mapping(
        self,
        omega: List[npt.NDArray[np.float64]],
        optimal_weights: npt.NDArray[np.float64],
    ) -> List[npt.NDArray[np.float64]]:
        """Henrickのマッピング関数を適用"""
        mapped_omega = []
        for w, d in zip(omega, optimal_weights):
            numerator = w * (d + d * d - 3 * d * w + w * w)
            denominator = d * d + w * (1 - 2 * d)
            mapped_omega.append(numerator / denominator)
        return mapped_omega

    def _borges_mapping(
        self,
        omega: List[npt.NDArray[np.float64]],
        optimal_weights: npt.NDArray[np.float64],
    ) -> List[npt.NDArray[np.float64]]:
        """Borgesのマッピング関数を適用"""
        mapped_omega = []
        for w, d in zip(omega, optimal_weights):
            mapped_w = (
                d * ((d + d - 3) * w * w + (3 - 2 * d) * w) / (d * d + w * (1 - 2 * d))
            )
            mapped_omega.append(mapped_w)
        return mapped_omega

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "epsilon": self._epsilon,
            "p": self._p,
            "cache_size": len(self._cache),
            "last_weights": self._cache.get("omega", None),
            "last_alpha": self._cache.get("alpha", None),
        }

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
