"""多相流体の界面計算機能を提供するモジュール

このモジュールは、多相流体における界面の初期化、再構築、幾何計算、
相識別などの機能を統合的に提供します。
"""

from typing import Tuple, List, Dict, Any
import numpy as np

from core.field import ScalarField, VectorField
from .operators.initialization import InitializationOperator
from .operators.reinitialization import ReinitializationOperator
from .operators.geometry import GeometryOperator
from .operators.indicator import IndicatorOperator


class InterfaceOperations:
    """多相流体の界面計算機能を提供するクラス"""

    def __init__(self, dx: np.ndarray, epsilon: float = 1.0e-6):
        """界面計算機能を初期化

        Args:
            dx: グリッド間隔（ベクトル）
            epsilon: 数値計算の安定化パラメータ
        """
        self.dx = dx
        self.epsilon = epsilon

        # 各演算子の初期化
        self._init_op = InitializationOperator(dx)
        self._reinit_op = ReinitializationOperator(dx, epsilon)
        self._geom_op = GeometryOperator(dx, epsilon)
        self._indicator_op = IndicatorOperator(epsilon)

    # 界面生成メソッド
    def create_sphere(
        self, shape: Tuple[int, ...], center: List[float], radius: float
    ) -> ScalarField:
        """球形の界面を生成"""
        return self._init_op.create_sphere(shape, center, radius)

    def create_plane(
        self, shape: Tuple[int, ...], normal: List[float], point: List[float]
    ) -> ScalarField:
        """平面界面を生成"""
        result = ScalarField(shape, self.dx)
        coords = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
        
        # ベクトル演算を活用した簡潔な実装
        result.data = np.dot(
            normal, 
            [(x - p) / d for x, p, d in zip(coords, point, self.dx)]
        )
        return result

    def combine_interfaces(
        self, phi1: ScalarField, phi2: ScalarField, operation: str = "union"
    ) -> ScalarField:
        """界面を組み合わせて新しい形状を生成"""
        return self._init_op.create_composite(phi1, phi2, operation)

    # 距離関数の再構築メソッド
    def reinitialize(
        self, phi: ScalarField, n_steps: int = 5, dt: float = 0.1
    ) -> ScalarField:
        """距離関数の性質を回復"""
        return self._reinit_op.execute(phi, n_steps, dt)

    def validate_distance(self, phi: ScalarField) -> float:
        """距離関数の性質を検証"""
        return self._reinit_op.validate(phi)

    # 幾何量の計算メソッド
    def compute_normal(self, phi: ScalarField) -> VectorField:
        """法線ベクトルを計算"""
        return self._geom_op.compute_normal(phi)

    def compute_curvature(
        self, phi: ScalarField, high_order: bool = False
    ) -> ScalarField:
        """曲率を計算"""
        method = "high_order" if high_order else "standard"
        return self._geom_op.compute_curvature(phi, method)

    # 相の分布計算メソッド
    def get_phase_distribution(self, phi: ScalarField) -> ScalarField:
        """相分布を計算（Heaviside関数）"""
        return self._indicator_op.compute_heaviside(phi)

    def get_interface_delta(self, phi: ScalarField) -> ScalarField:
        """界面のデルタ関数を計算"""
        return self._indicator_op.compute_delta(phi)

    def get_property_field(
        self, phi: ScalarField, value1: float, value2: float
    ) -> ScalarField:
        """物性値の空間分布を計算"""
        return self._indicator_op.get_phase_field(phi, value1, value2)

    # 診断情報の取得メソッド
    def get_diagnostics(self, phi: ScalarField) -> Dict[str, Any]:
        """界面に関する診断情報を取得"""
        # 相分布とデルタ関数の計算
        phase = self.get_phase_distribution(phi)
        delta = self.get_interface_delta(phi)

        # 距離関数の性質の検証
        distance_error = self.validate_distance(phi)

        # 界面の曲率を計算（標準精度で十分）
        kappa = self.compute_curvature(phi, high_order=False)

        # グリッド体積要素を計算
        dv = np.prod(self.dx)

        # 界面近傍の点のみを考慮
        min_dx = np.min(self.dx)
        interface_region = np.abs(phi.data) < (5.0 * min_dx)

        # 界面上の曲率統計の計算
        if np.any(interface_region):
            kappa_interface = kappa.data[interface_region]
            kappa_min = float(np.min(kappa_interface))
            kappa_max = float(np.max(kappa_interface))
            kappa_mean = float(np.mean(kappa_interface))
            kappa_rms = float(np.sqrt(np.mean(kappa_interface**2)))
        else:
            kappa_min = kappa_max = kappa_mean = kappa_rms = 0.0

        # 診断情報の辞書を構築
        return {
            "volume_fraction": float(np.mean(phase.data)),
            "interface_points": int(np.sum(interface_region)),
            "interface_area": float(np.sum(delta.data) * dv),
            "distance_error": distance_error,
            "curvature": {
                "min": kappa_min,
                "max": kappa_max,
                "mean": kappa_mean,
                "rms": kappa_rms,
            },
        }

    def compute_interface_measures(self, phi: ScalarField) -> Dict[str, float]:
        """界面の各種測度を計算"""
        # 相分布とデルタ関数
        phase = self.get_phase_distribution(phi)
        delta = self.get_interface_delta(phi)

        # グリッド体積要素の計算
        dv = np.prod(self.dx)

        # 体積の計算
        volume = float(np.sum(phase.data) * dv)

        # 界面の面積/長さの計算
        interface_measure = float(np.sum(delta.data) * dv)

        return {
            "volume": volume,
            "area": interface_measure if phi.ndim == 3 else None,
            "length": interface_measure if phi.ndim == 2 else None,
            "perimeter": interface_measure if phi.ndim == 3 else None,
        }