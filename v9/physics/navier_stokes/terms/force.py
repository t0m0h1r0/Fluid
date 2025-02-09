"""Navier-Stokes方程式の外力項を提供するモジュール

このモジュールは、重力や表面張力などの外力の効果を実装します。
"""

from typing import List, Dict, Any, Optional
import numpy as np
from core.field import VectorField
from ..base import NavierStokesTerm
from physics.levelset import LevelSetField


class ForceBase:
    """外力の基底クラス"""

    def __init__(self, name: str):
        """外力を初期化

        Args:
            name: 外力の名前
        """
        self.name = name
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """外力が有効かどうかを取得"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """外力の有効/無効を設定"""
        self._enabled = value

    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        """外力の寄与を計算"""
        raise NotImplementedError

    def get_diagnostics(self, velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """外力の診断情報を取得"""
        raise NotImplementedError


class GravityForce(ForceBase):
    """重力項"""

    def __init__(self, gravity: float = 9.81, direction: int = -1):
        """重力項を初期化

        Args:
            gravity: 重力加速度
            direction: 重力方向の軸インデックス（負値は逆方向）
        """
        super().__init__(name="Gravity")
        self.gravity = gravity
        self.direction = direction

    def compute(
        self,
        velocity: VectorField,
        levelset: Optional[LevelSetField] = None,
        properties=None,
        **kwargs,
    ) -> List[np.ndarray]:
        """重力項の寄与を計算"""
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        result = [np.zeros_like(v.data) for v in velocity.components]
        axis = abs(self.direction) % velocity.ndim
        sign = -1 if self.direction < 0 else 1

        # 浮力効果の計算
        if levelset is not None and properties is not None:
            density = properties.get_density(levelset)
            rho_ref = properties.get_reference_density()
            if rho_ref is not None and rho_ref > 0:
                result[axis] = sign * self.gravity * (density.data / rho_ref - 1.0)
        else:
            # 単純な重力加速度
            result[axis] = sign * self.gravity

        return result

    def get_diagnostics(self, velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """重力項の診断情報を取得"""
        return {"type": "gravity", "gravity": self.gravity, "direction": self.direction}


class SurfaceTensionForce(ForceBase):
    """表面張力項"""

    def __init__(self, surface_tension: float = 0.07):
        """表面張力項を初期化

        Args:
            surface_tension: 表面張力係数（N/m）
        """
        super().__init__(name="SurfaceTension")
        self.surface_tension = surface_tension

    def compute(
        self,
        velocity: VectorField,
        levelset: Optional[LevelSetField] = None,
        properties=None,
        **kwargs,
    ) -> List[np.ndarray]:
        """表面張力項の寄与を計算"""
        if not self.enabled or levelset is None:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 表面張力係数の取得（プロパティから、またはデフォルト値）
        sigma = (
            properties.get_surface_tension_coefficient()
            if properties is not None
            else self.surface_tension
        )

        if sigma == 0:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 界面の法線と曲率を計算
        kappa = levelset.curvature()
        delta = levelset.delta()

        # 表面力の計算
        force = sigma * kappa * delta

        # 密度による重みづけ（オプション）
        if properties is not None:
            density = properties.get_density(levelset)
            force /= density.data

        # 各方向の力を計算
        result = [np.zeros_like(v.data) for v in velocity.components]
        for i in range(velocity.ndim):
            # 法線方向の力を計算
            grad_phi = levelset.gradient(i)
            grad_norm = np.sqrt(
                sum(levelset.gradient(j) ** 2 for j in range(velocity.ndim))
            )
            normal = grad_phi / (grad_norm + 1e-10)  # ゼロ除算防止
            result[i] = force * normal

        return result

    def get_diagnostics(
        self, velocity: VectorField, levelset: Optional[LevelSetField] = None, **kwargs
    ) -> Dict[str, Any]:
        """表面張力項の診断情報を取得"""
        diag = {
            "type": "surface_tension",
            "coefficient": self.surface_tension,
        }

        if levelset is not None:
            kappa = levelset.curvature()
            diag.update(
                {
                    "max_curvature": float(np.max(np.abs(kappa))),
                    "interface_length": float(np.sum(levelset.delta()))
                    * velocity.dx**velocity.ndim,
                }
            )

        return diag


class ForceTerm(NavierStokesTerm):
    """外力項クラス"""

    def __init__(self, forces: Optional[List[ForceBase]] = None):
        """外力項を初期化

        Args:
            forces: 外力のリスト
        """
        self._name = "Force"
        self.forces = forces or []

    @property
    def name(self) -> str:
        """項の名前を取得"""
        return self._name

    def compute(self, velocity: VectorField, dt: float, **kwargs) -> List[np.ndarray]:
        """外力項の寄与を計算"""
        if not self.forces:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 各外力の寄与を合計
        result = [np.zeros_like(v.data) for v in velocity.components]
        for force in self.forces:
            if force.enabled:
                force_contribution = force.compute(velocity, **kwargs)
                for i in range(len(result)):
                    result[i] += force_contribution[i]

        return result

    def get_diagnostics(self, velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """外力項の診断情報を取得"""
        diag = {"enabled_forces": len([f for f in self.forces if f.enabled])}

        # 各外力の診断情報を収集
        for force in self.forces:
            if force.enabled:
                diag[force.name] = force.get_diagnostics(velocity, **kwargs)

        return diag
