"""Navier-Stokes方程式の外力項を提供するモジュール

このモジュールは、Navier-Stokes方程式の外力項（重力、表面張力など）を実装します。
複数の外力を統一的に扱い、特に表面張力についてはLevel Set法との連携を考慮します。
"""

import numpy as np
from typing import List, Dict, Any, Optional
from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from ..base_term import NavierStokesTerm


class ExternalForce:
    """外力の基底クラス

    個々の外力（重力、表面張力など）の計算を行います。
    """

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


class GravityForce(ExternalForce):
    """重力項

    重力加速度による外力を計算します。密度変化がある場合は浮力も考慮します。
    """

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
        density: Optional[ScalarField] = None,
        reference_density: Optional[float] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """重力項の寄与を計算

        Args:
            velocity: 現在の速度場
            density: 密度場（Noneの場合は一様密度を仮定）
            reference_density: 浮力計算の基準密度
            **kwargs: 未使用

        Returns:
            各方向の速度成分への寄与のリスト
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        result = [np.zeros_like(v.data) for v in velocity.components]
        axis = abs(self.direction) % velocity.ndim
        sign = -1 if self.direction < 0 else 1

        if density is None or reference_density is None:
            # 単純な重力加速度
            result[axis] = sign * self.gravity
        else:
            # 浮力を考慮した実効的な重力加速度
            result[axis] = (
                sign * self.gravity * (density.data / reference_density - 1.0)
            )

        return result

    def get_diagnostics(self, velocity: VectorField, **kwargs) -> Dict[str, Any]:
        """重力項の診断情報を取得"""
        return {"type": "gravity", "gravity": self.gravity, "direction": self.direction}


class SurfaceTensionForce(ExternalForce):
    """表面張力項

    Level Set法で表現された界面での表面張力を計算します。
    """

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
        levelset: LevelSetField,
        density: Optional[ScalarField] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """表面張力項の寄与を計算

        Args:
            velocity: 現在の速度場
            levelset: Level Set場
            density: 密度場（Noneの場合は一様密度を仮定）
            **kwargs: 未使用

        Returns:
            各方向の速度成分への寄与のリスト
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 界面の法線と曲率を計算
        kappa = levelset.curvature()
        grad_phi = np.array([levelset.gradient(i) for i in range(velocity.ndim)])
        grad_norm = np.sqrt(np.sum(grad_phi**2, axis=0))
        grad_norm = np.maximum(grad_norm, 1e-10)  # ゼロ除算を防ぐ
        normal = grad_phi / grad_norm

        # 表面力の計算
        surface_force = self.surface_tension * kappa * levelset.delta()

        # 密度の考慮
        if density is not None:
            surface_force /= density.data

        # 各方向の力を計算
        result = []
        for n in normal:
            result.append(surface_force * n)

        return result

    def get_diagnostics(
        self, velocity: VectorField, levelset: LevelSetField, **kwargs
    ) -> Dict[str, Any]:
        """表面張力項の診断情報を取得"""
        kappa = levelset.curvature()
        return {
            "type": "surface_tension",
            "coefficient": self.surface_tension,
            "max_curvature": np.max(np.abs(kappa)),
            "interface_length": np.sum(levelset.delta()) * velocity.dx**velocity.ndim,
        }


class ForceTerm(NavierStokesTerm):
    """外力項クラス

    Navier-Stokes方程式の外力項を管理し、複数の外力の寄与を合計します。
    """

    def __init__(self):
        """外力項を初期化"""
        super().__init__(name="Force")
        self.forces: List[ExternalForce] = []

    def add_force(self, force: ExternalForce):
        """外力を追加

        Args:
            force: 追加する外力
        """
        self.forces.append(force)

    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]:
        """全ての外力の寄与を計算

        Args:
            velocity: 現在の速度場
            **kwargs: 各外力に必要なパラメータ

        Returns:
            各方向の速度成分への寄与のリスト
        """
        if not self.enabled or not self.forces:
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
        diag = super().get_diagnostics(velocity, **kwargs)

        # 各外力の診断情報を収集
        force_diag = {}
        for force in self.forces:
            if force.enabled:
                force_diag[force.name] = force.get_diagnostics(velocity, **kwargs)

        diag["forces"] = force_diag
        return diag
