"""WENOスキームの重み係数を計算するモジュール（改良版）"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import numpy.typing as npt
from core.field import ScalarField


class WeightCalculator:
    """WENOスキームの重み係数を計算するクラス（改良版）"""

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
        """非線形重み係数を計算（改良版）

        Args:
            beta: 各ステンシルの滑らかさ指標
            optimal_weights: 理想重み係数

        Returns:
            (非線形重み係数のリスト, 正規化係数）のタプル
        """
        # ScalarFieldを活用した計算
        alpha = []
        for b, d in zip(beta, optimal_weights):
            beta_field = ScalarField(b.shape, np.ones(len(b.shape)), initial_value=b)
            # 新しい演算子を活用した計算
            alpha_k = d / ((self._epsilon + beta_field) ** self._p)
            alpha.append(alpha_k.data)

        # 正規化係数の計算（新しい演算子を活用）
        omega_sum = sum(a for a in alpha)

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
        """マッピング関数を使用して非線形重み係数を計算（改良版）

        Args:
            beta: 各ステンシルの滑らかさ指標
            optimal_weights: 理想重み係数
            mapping_function: マッピング関数の種類

        Returns:
            (非線形重み係数のリスト, 正規化係数）のタプル
        """
        # まず通常の重み係数を計算
        omega, omega_sum = self.compute_weights(beta, optimal_weights)

        if mapping_function is None:
            return omega, omega_sum

        # マッピング関数の適用（新しい演算子を活用）
        if mapping_function == "henrick":
            omega = self._henrick_mapping(omega, optimal_weights)
        elif mapping_function == "borges":
            omega = self._borges_mapping(omega, optimal_weights)
        else:
            raise ValueError(f"未知のマッピング関数です: {mapping_function}")

        # 正規化係数の再計算
        omega_sum = sum(o for o in omega)

        return omega, omega_sum

    def _henrick_mapping(
        self,
        omega: List[npt.NDArray[np.float64]],
        optimal_weights: npt.NDArray[np.float64],
    ) -> List[npt.NDArray[np.float64]]:
        """Henrickのマッピング関数を適用（改良版）"""
        mapped_omega = []
        for w, d in zip(omega, optimal_weights):
            # ScalarFieldを活用した計算
            w_field = ScalarField(w.shape, np.ones(len(w.shape)), initial_value=w)
            numerator = w_field * (d + d * d - 3 * d * w_field + w_field * w_field)
            denominator = d * d + w_field * (1 - 2 * d)
            mapped_omega.append((numerator / denominator).data)
        return mapped_omega

    def _borges_mapping(
        self,
        omega: List[npt.NDArray[np.float64]],
        optimal_weights: npt.NDArray[np.float64],
    ) -> List[npt.NDArray[np.float64]]:
        """Borgesのマッピング関数を適用（改良版）"""
        mapped_omega = []
        for w, d in zip(omega, optimal_weights):
            # ScalarFieldを活用した計算
            w_field = ScalarField(w.shape, np.ones(len(w.shape)), initial_value=w)
            numerator = d * ((d + d - 3) * w_field * w_field + (3 - 2 * d) * w_field)
            denominator = d * d + w_field * (1 - 2 * d)
            mapped_omega.append((numerator / denominator).data)
        return mapped_omega

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得（改良版）"""
        return {
            "epsilon": self._epsilon,
            "p": self._p,
            "cache_size": len(self._cache),
            "last_weights": self._cache.get("omega"),
            "last_alpha": self._cache.get("alpha"),
            "mapping_stats": {
                "min_weight": min(np.min(w) for w in self._cache.get("omega", [0]))
                if "omega" in self._cache
                else None,
                "max_weight": max(np.max(w) for w in self._cache.get("omega", [0]))
                if "omega" in self._cache
                else None,
            },
        }

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
