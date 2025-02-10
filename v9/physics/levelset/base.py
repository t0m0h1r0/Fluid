"""Level Set法の基底クラスとプロトコルを定義"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional
import numpy as np


class LevelSetTerm(Protocol):
    """Level Set方程式の項のプロトコル"""

    @property
    def name(self) -> str:
        """項の名前"""
        ...

    def compute(
        self,
        levelset: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        dt: float = 0.0,
        **kwargs,
    ) -> np.ndarray:
        """Level Set場への寄与を計算"""
        ...

    def get_diagnostics(
        self, levelset: np.ndarray, velocity: Optional[np.ndarray] = None, **kwargs
    ) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


class LevelSetTermBase:
    """Level Set項の基底実装"""

    def __init__(self, name: str, enabled: bool = True):
        """基底クラスを初期化

        Args:
            name: 項の名前
            enabled: 項が有効かどうか
        """
        self._name = name
        self._enabled = enabled
        self._diagnostics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """項の名前を取得"""
        return self._name

    @property
    def enabled(self) -> bool:
        """項が有効かどうかを取得"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """項の有効/無効を設定"""
        self._enabled = value

    def get_diagnostics(
        self, levelset: np.ndarray, velocity: Optional[np.ndarray] = None, **kwargs
    ) -> Dict[str, Any]:
        """診断情報を取得"""
        return {"name": self.name, "enabled": self.enabled, **self._diagnostics}

    def update_diagnostics(self, **kwargs):
        """診断情報を更新"""
        self._diagnostics.update(kwargs)


class LevelSetSolverBase(ABC):
    """Level Set ソルバーの基底抽象クラス"""

    def __init__(self, use_weno: bool = True, weno_order: int = 5, logger=None):
        """基底ソルバーを初期化

        Args:
            use_weno: WENOスキームを使用するかどうか
            weno_order: WENOスキームの次数
            logger: ロガー
        """
        self.use_weno = use_weno
        self.weno_order = weno_order
        self.logger = logger

        # 計算状態の追跡
        self._time = 0.0
        self._dt = None
        self._iteration_count = 0

    @abstractmethod
    def compute_derivative(self, state: Any, **kwargs) -> np.ndarray:
        """Level Set関数の時間微分を計算"""
        pass

    @abstractmethod
    def compute_timestep(self, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算"""
        pass

    def _compute_weno_reconstruction(self, values: np.ndarray, axis: int) -> np.ndarray:
        """WENOスキームによる再構築の共通実装

        Args:
            values: 再構築する値の配列
            axis: 再構築を行う軸

        Returns:
            再構築された値
        """
        # WENO5の実装
        if self.weno_order == 5:
            v1 = np.roll(values, 2, axis=axis)
            v2 = np.roll(values, 1, axis=axis)
            v3 = values
            v4 = np.roll(values, -1, axis=axis)
            v5 = np.roll(values, -2, axis=axis)

            # 各ステンシルでの滑らかさ指標を計算
            eps = 1e-6  # ゼロ除算防止用
            beta0 = (
                13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 1 / 4 * (v1 - 4 * v2 + 3 * v3) ** 2
            )
            beta1 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 1 / 4 * (v2 - v4) ** 2
            beta2 = (
                13 / 12 * (v3 - 2 * v4 + v5) ** 2 + 1 / 4 * (3 * v3 - 4 * v4 + v5) ** 2
            )

            # 非線形重みを計算
            weights = np.array([0.1, 0.6, 0.3])
            alpha0 = weights[0] / (eps + beta0) ** 2
            alpha1 = weights[1] / (eps + beta1) ** 2
            alpha2 = weights[2] / (eps + beta2) ** 2
            alpha_sum = alpha0 + alpha1 + alpha2

            omega0 = alpha0 / alpha_sum
            omega1 = alpha1 / alpha_sum
            omega2 = alpha2 / alpha_sum

            # 各ステンシルでの補間値を計算
            weno_coeffs = [
                [1 / 3, -7 / 6, 11 / 6],  # 左側ステンシル
                [-1 / 6, 5 / 6, 1 / 3],  # 中央ステンシル
                [1 / 3, 5 / 6, -1 / 6],  # 右側ステンシル
            ]

            p0 = (
                weno_coeffs[0][0] * v1 + weno_coeffs[0][1] * v2 + weno_coeffs[0][2] * v3
            )
            p1 = (
                weno_coeffs[1][0] * v2 + weno_coeffs[1][1] * v3 + weno_coeffs[1][2] * v4
            )
            p2 = (
                weno_coeffs[2][0] * v3 + weno_coeffs[2][1] * v4 + weno_coeffs[2][2] * v5
            )

            return omega0 * p0 + omega1 * p1 + omega2 * p2

        # WENO3の実装
        elif self.weno_order == 3:
            v1 = np.roll(values, 1, axis=axis)
            v2 = values
            v3 = np.roll(values, -1, axis=axis)

            # 各ステンシルでの滑らかさ指標を計算
            beta0 = (v2 - v1) ** 2
            beta1 = (v3 - v2) ** 2

            eps = 1e-6
            weights = np.array([1 / 3, 2 / 3])
            alpha0 = weights[0] / (eps + beta0) ** 2
            alpha1 = weights[1] / (eps + beta1) ** 2
            alpha_sum = alpha0 + alpha1

            omega0 = alpha0 / alpha_sum
            omega1 = alpha1 / alpha_sum

            weno_coeffs = [
                [-1 / 2, 3 / 2],  # 左側ステンシル
                [1 / 2, 1 / 2],  # 右側ステンシル
            ]

            p0 = weno_coeffs[0][0] * v1 + weno_coeffs[0][1] * v2
            p1 = weno_coeffs[1][0] * v2 + weno_coeffs[1][1] * v3

            return omega0 * p0 + omega1 * p1

        else:
            raise ValueError(f"未対応のWENO次数です: {self.weno_order}")
