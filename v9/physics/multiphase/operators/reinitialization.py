"""界面の再構築を提供するモジュール

このモジュールは、界面の位置を保持したまま符号付き距離関数の
性質を回復する再構築機能を提供します。
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from core.field import ScalarField


class ReinitializationOperator:
    """距離関数の再構築を実行するクラス"""

    def __init__(self, dx: float, epsilon: float = 1.0e-6):
        """再構築演算子を初期化
        
        Args:
            dx: グリッド間隔
            epsilon: 数値計算の安定化パラメータ
        """
        self.dx = dx
        self.epsilon = epsilon

    def execute(
        self, phi: ScalarField, n_steps: int = 5, dt: float = 0.1
    ) -> ScalarField:
        """距離関数の性質を回復
        
        高速行進法を用いて以下の発展方程式を解きます：
        ∂φ/∂τ = sign(φ₀)(1 - |∇φ|)
        
        Args:
            phi: 入力スカラー場
            n_steps: 反復回数
            dt: 疑似時間の時間刻み幅
            
        Returns:
            再構築されたスカラー場
        """
        # 結果の初期化
        result = phi.copy()
        sign = np.sign(result.data)

        # 高速行進法による再構築
        for _ in range(n_steps):
            # 勾配を計算
            grad = np.array([result.gradient(i) for i in range(result.ndim)])
            grad_norm = np.sqrt(np.sum(grad**2, axis=0))
            grad_norm = np.maximum(grad_norm, self.epsilon)  # 安定化

            # 時間発展
            correction = sign * (grad_norm - 1.0)
            result.data -= dt * correction

            # 数値的安定化のためにガウシアンフィルタを適用
            result.data = gaussian_filter(result.data, sigma=0.5 * self.dx)

        return result

    def validate(self, phi: ScalarField) -> float:
        """距離関数の性質を検証
        
        |∇φ| = 1 からのずれを計算します。
        
        Args:
            phi: 検証するスカラー場
            
        Returns:
            勾配ノルムの1からの平均二乗偏差
        """
        # 勾配ノルムの計算
        grad = np.array([phi.gradient(i) for i in range(phi.ndim)])
        grad_norm = np.sqrt(np.sum(grad**2, axis=0))

        # 界面近傍での |∇φ| = 1 からのずれを計算
        interface_region = np.abs(phi.data) < (5.0 * self.dx)
        if np.any(interface_region):
            deviation = np.mean((grad_norm[interface_region] - 1.0) ** 2)
            return float(deviation)
        return 0.0

    def _compute_stable_sign(self, phi: ScalarField, width: float = 5.0) -> np.ndarray:
        """安定化された符号関数を計算
        
        Args:
            phi: スカラー場
            width: 遷移領域の幅（グリッド間隔の倍数）
            
        Returns:
            安定化された符号関数の値
        """
        return phi.data / np.sqrt(phi.data**2 + (width * self.dx)**2)