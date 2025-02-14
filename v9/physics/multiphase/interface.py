"""多相流体の界面計算機能を提供するモジュール（改良版）

このモジュールは、多相流体における界面の初期化、再構築、幾何計算、
相識別などの機能を統合的に提供します。新しい演算子とメソッドを活用して
実装を改善しています。
"""

from typing import Tuple, List, Dict, Any
import numpy as np

from core.field import ScalarField, VectorField
from .operators.initialization import InitializationOperator
from .operators.reinitialization import ReinitializationOperator
from .operators.geometry import GeometryOperator
from .operators.indicator import IndicatorOperator


class InterfaceOperations:
    """多相流体の界面計算機能を提供するクラス（改良版）"""

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

    def create_sphere(
        self, shape: Tuple[int, ...], center: List[float], radius: float
    ) -> ScalarField:
        """球形の界面を生成（新しい演算子を活用）"""
        return self._init_op.create_sphere(shape, center, radius)

    def create_plane(
        self, shape: Tuple[int, ...], normal: List[float], point: List[float]
    ) -> ScalarField:
        """平面界面を生成（新しい演算子を活用）"""
        # 座標場の生成
        coords = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")

        # ベクトル演算を活用した効率的な実装
        normal = np.array(normal) / np.linalg.norm(normal)
        displacement = [x - p for x, p in zip(coords, point)]
        return ScalarField(
            shape,
            self.dx,
            initial_value=sum(n * d for n, d in zip(normal, displacement)),
        )

    def compute_normal(self, phi: ScalarField) -> VectorField:
        """法線ベクトルを計算（新しい演算子を活用）"""
        # 勾配の計算と正規化
        grad = phi.gradient()
        grad_norm = grad.magnitude()
        # 新しい除算演算子を使用
        return grad / (grad_norm + self.epsilon)

    def compute_curvature(
        self, phi: ScalarField, high_order: bool = False
    ) -> ScalarField:
        """曲率を計算（新しい演算子を活用）"""
        method = "high_order" if high_order else "standard"
        return self._geom_op.compute_curvature(phi, method)

    def combine_interfaces(
        self, phi1: ScalarField, phi2: ScalarField, operation: str = "union"
    ) -> ScalarField:
        """界面を組み合わせて新しい形状を生成（新しい演算子を活用）"""
        # 同じ形状チェック
        if phi1.shape != phi2.shape:
            raise ValueError("スカラー場の形状が一致しません")

        result = ScalarField(phi1.shape, self.dx)

        if operation == "union":
            # minimum演算子を使用
            result.data = np.minimum(phi1.data, phi2.data)
        elif operation == "intersection":
            # maximum演算子を使用
            result.data = np.maximum(phi1.data, phi2.data)
        elif operation == "difference":
            # 新しい演算子を使用
            result.data = np.maximum(phi1.data, -phi2.data)
        else:
            raise ValueError(f"未知の操作: {operation}")

        return result

    def get_phase_distribution(self, phi: ScalarField) -> ScalarField:
        """相分布を計算（新しい演算子を活用）"""
        return self._indicator_op.compute_heaviside(phi)

    def get_interface_delta(self, phi: ScalarField) -> ScalarField:
        """界面のデルタ関数を計算（新しい演算子を活用）"""
        return self._indicator_op.compute_delta(phi)

    def get_property_field(
        self, phi: ScalarField, value1: float, value2: float
    ) -> ScalarField:
        """物性値の空間分布を計算（新しい演算子を活用）"""
        return self._indicator_op.get_phase_field(phi, value1, value2)

    def _compute_interface_curvature_stats(
        self, kappa: ScalarField, interface_region: np.ndarray
    ) -> Dict[str, float]:
        """界面上の曲率統計を計算（新しいメソッドを活用）"""
        if not np.any(interface_region):
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "rms": 0.0}

        kappa_interface = kappa.data[interface_region]
        return {
            "min": float(np.min(kappa_interface)),
            "max": float(np.max(kappa_interface)),
            "mean": float(np.mean(kappa_interface)),
            "rms": float(np.sqrt(np.mean(kappa_interface**2))),
        }

    def get_diagnostics(self, phi: ScalarField) -> Dict[str, Any]:
        """界面に関する診断情報を取得（新しいメソッドを活用）"""
        # 相分布とデルタ関数の計算
        phase = self.get_phase_distribution(phi)
        delta = self.get_interface_delta(phi)

        # 距離関数の性質の検証
        distance_error = self._reinit_op.validate(phi)

        # 界面の曲率を計算
        kappa = self.compute_curvature(phi, high_order=False)

        # グリッド体積要素の計算
        dv = np.prod(self.dx)

        # 界面近傍の点の識別（新しいメソッドを活用）
        interface_region = np.abs(phi.data) < (5.0 * min(self.dx))

        # 界面上の曲率統計の計算
        curvature_stats = self._compute_interface_curvature_stats(
            kappa, interface_region
        )

        return {
            "volume_fraction": float(phase.mean()),
            "interface_points": int(interface_region.sum()),
            "interface_area": float((delta.data * dv).sum()),
            "distance_error": distance_error,
            "curvature": curvature_stats,
        }

    def compute_interface_measures(self, phi: ScalarField) -> Dict[str, float]:
        """界面の各種測度を計算（新しいメソッドを活用）"""
        # 相分布とデルタ関数の計算
        phase = self.get_phase_distribution(phi)
        delta = self.get_interface_delta(phi)

        # グリッド体積要素の計算
        dv = np.prod(self.dx)

        # 体積の計算（新しい積分メソッドを使用）
        volume = float(phase.integrate())

        # 界面測度の計算
        interface_measure = float(delta.integrate())

        return {
            "volume": volume,
            "area": interface_measure if phi.ndim == 3 else None,
            "length": interface_measure if phi.ndim == 2 else None,
            "perimeter": interface_measure if phi.ndim == 3 else None,
        }

    def reinitialize(
        self, phi: ScalarField, n_steps: int = 5, dt: float = 0.1
    ) -> ScalarField:
        """距離関数の性質を回復（新しいメソッドを活用）"""
        return self._reinit_op.execute(phi, n_steps, dt)

    def validate_distance(self, phi: ScalarField) -> float:
        """距離関数の性質を検証（新しいメソッドを活用）"""
        return self._reinit_op.validate(phi)
