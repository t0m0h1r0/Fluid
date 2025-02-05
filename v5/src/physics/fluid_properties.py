import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class FluidProperties:
    """流体物性値を保持するデータクラス"""

    name: str
    density: float
    viscosity: float


class MultiPhaseProperties:
    """
    多相流体の物性値を管理するクラス

    異なる相の物性値を補間し、混相流れを扱う
    """

    def __init__(self, fluids: Dict[str, FluidProperties]):
        """
        コンストラクタ

        Args:
            fluids (Dict[str, FluidProperties]): 流体の物性値辞書
        """
        self.fluids = fluids

    def get_density(self, phase_indicator: np.ndarray) -> np.ndarray:
        """
        位相指示関数から密度を計算

        Args:
            phase_indicator (np.ndarray): 位相指示関数

        Returns:
            np.ndarray: 補間された密度場
        """
        # 密度値の配列を取得
        densities = np.array([fluid.density for fluid in self.fluids.values()])

        # 安全な補間
        interpolated = self._safe_interpolate(densities, phase_indicator)

        # 最小値制限
        min_density = 1e-10
        return np.maximum(interpolated, min_density)

    def get_viscosity(self, phase_indicator: np.ndarray) -> np.ndarray:
        """
        位相指示関数から粘性係数を計算

        Args:
            phase_indicator (np.ndarray): 位相指示関数

        Returns:
            np.ndarray: 補間された粘性係数場
        """
        # 粘性係数の配列を取得
        viscosities = np.array([fluid.viscosity for fluid in self.fluids.values()])

        # 安全な補間
        interpolated = self._safe_interpolate(viscosities, phase_indicator)

        # 最小値制限
        min_viscosity = 1e-10
        return np.maximum(interpolated, min_viscosity)

    def _safe_interpolate(
        self,
        properties: np.ndarray,
        phase_indicator: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        """
        安全な線形補間

        Args:
            properties (np.ndarray): 物性値の配列
            phase_indicator (np.ndarray): 位相指示関数
            method (str): 補間方法（現在は線形補間のみ）

        Returns:
            np.ndarray: 補間された物性値場
        """
        if method == "linear":
            # ヘビサイド関数による重み付け
            weights = self._heaviside(phase_indicator)

            # 安全な重み付け補間
            interpolated = np.sum(
                properties[:, np.newaxis, np.newaxis, np.newaxis] * weights, axis=0
            )

            return interpolated
        else:
            raise ValueError(f"未サポートの補間方法: {method}")

    def _heaviside(self, phi: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """
        スムーズなヘビサイド関数

        Args:
            phi (np.ndarray): 位相場
            epsilon (float): インターフェース厚さ

        Returns:
            np.ndarray: 重み係数
        """
        return 0.5 * (1.0 + np.tanh(phi / epsilon))
