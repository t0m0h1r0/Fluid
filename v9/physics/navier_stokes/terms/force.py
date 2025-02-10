from typing import List, Dict, Any, Optional
import numpy as np

from core.field import VectorField, ScalarField
from .base import BaseNavierStokesTerm


class GravityForce(BaseNavierStokesTerm):
    """重力項クラス"""

    def __init__(
        self,
        gravity: float = 9.81,
        direction: int = -1,
        name: str = "Gravity",
        enabled: bool = True,
    ):
        """
        Args:
            gravity: 重力加速度
            direction: 重力方向の軸インデックス（負値は逆方向）
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)
        self._gravity = gravity
        self._direction = direction

    def compute(
        self, velocity: VectorField, density: Optional[ScalarField] = None, **kwargs
    ) -> List[np.ndarray]:
        """重力項の寄与を計算

        Args:
            velocity: 速度場
            density: 密度場（オプション）

        Returns:
            各方向の速度成分への重力項の寄与
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        result = [np.zeros_like(v.data) for v in velocity.components]
        axis = abs(self._direction) % velocity.ndim
        sign = -1 if self._direction < 0 else 1

        # 密度場が提供されている場合は密度に応じた重力
        if density is not None:
            result[axis] = sign * self._gravity * density.data
        else:
            # デフォルトは一様な重力
            result[axis].fill(sign * self._gravity)

        # 診断情報の更新
        self._diagnostics = {
            "gravity": self._gravity,
            "direction": self._direction,
            "max_force": float(np.max(np.abs(result[axis]))),
        }

        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "gravity": self._gravity,
                "direction": self._direction,
                "max_force": self._diagnostics.get("max_force", 0.0),
            }
        )
        return diag


class SurfaceTensionForce(BaseNavierStokesTerm):
    """表面張力項クラス"""

    def __init__(
        self,
        surface_tension: float = 0.072,
        name: str = "SurfaceTension",
        enabled: bool = True,
    ):
        """
        Args:
            surface_tension: 表面張力係数
            name: 項の名前
            enabled: 項を有効にするかどうか
        """
        super().__init__(name, enabled)
        self._surface_tension = surface_tension

    def compute(
        self, velocity: VectorField, levelset: ScalarField, **kwargs
    ) -> List[np.ndarray]:
        """表面張力項の寄与を計算

        Args:
            velocity: 速度場
            levelset: レベルセット関数場

        Returns:
            各方向の速度成分への表面張力項の寄与
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        result = [np.zeros_like(v.data) for v in velocity.components]
        dx = velocity.dx

        # 界面の法線ベクトルと曲率の計算
        grad_norm = 1e-10  # ゼロ除算防止
        grad_phi = []
        for i in range(velocity.ndim):
            g = np.gradient(levelset.data, dx, axis=i)
            grad_phi.append(g)
            grad_norm += g**2
        grad_norm = np.sqrt(grad_norm)

        # 曲率の計算
        kappa = sum(
            np.gradient(g / grad_norm, dx, axis=i) for i, g in enumerate(grad_phi)
        )

        # デルタ関数の計算（界面の特徴づけ）
        epsilon = kwargs.get("epsilon", 1e-2)
        delta = 0.5 / epsilon * (1.0 - np.tanh(levelset.data / epsilon) ** 2)

        # 表面張力の計算
        force = self._surface_tension * kappa * delta
        for i in range(velocity.ndim):
            result[i] = force * grad_phi[i] / grad_norm

        # 診断情報の更新
        self._diagnostics = {
            "surface_tension": self._surface_tension,
            "max_curvature": float(np.max(np.abs(kappa))),
            "max_force": float(np.max(np.abs(force))),
        }

        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        diag = super().get_diagnostics()
        diag.update(
            {
                "surface_tension": self._surface_tension,
                "max_curvature": self._diagnostics.get("max_curvature", 0.0),
                "max_force": self._diagnostics.get("max_force", 0.0),
            }
        )
        return diag
