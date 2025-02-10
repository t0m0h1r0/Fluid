"""Navier-Stokes方程式の外力項を実装するモジュール

このモジュールは、重力、浮力、表面張力などの外力の効果を実装します。
"""

from typing import List, Dict, Any, Optional, Protocol
import numpy as np

from core.field import VectorField
from physics.levelset import LevelSetField
from physics.levelset.properties import PropertiesManager
from .base import TermBase


class ForceComponent(Protocol):
    """外力成分のインターフェース"""

    @property
    def name(self) -> str:
        """外力の名前"""
        ...

    @property
    def enabled(self) -> bool:
        """外力が有効かどうか"""
        ...

    def compute(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> List[np.ndarray]:
        """外力の寄与を計算"""
        ...

    def get_diagnostics(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> Dict[str, Any]:
        """診断情報を取得"""
        ...


class GravityForce:
    """重力項クラス"""

    def __init__(self, gravity: float = 9.81, direction: int = -1):
        """重力項を初期化

        Args:
            gravity: 重力加速度
            direction: 重力方向の軸インデックス（負値は逆方向）
        """
        self._name = "Gravity"
        self._enabled = True
        self.gravity = gravity
        self.direction = direction

    @property
    def name(self) -> str:
        """外力の名前を取得"""
        return self._name

    @property
    def enabled(self) -> bool:
        """外力が有効かどうかを取得"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """外力の有効/無効を設定"""
        self._enabled = value

    def compute(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
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
            # 密度場とリファレンス密度を取得
            density = np.array(properties.get_density(levelset))
            rho_ref = properties.get_reference_density()

            if rho_ref is not None and rho_ref > 0:
                # 浮力の計算 (ρ/ρref - 1) * g
                result[axis] = sign * self.gravity * (density / rho_ref - 1.0)
        else:
            # 単純な重力加速度
            result[axis] = sign * self.gravity

        return result

    def get_diagnostics(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> Dict[str, Any]:
        """診断情報を取得"""
        return {
            "type": "gravity",
            "gravity": self.gravity,
            "direction": self.direction,
            "reference_density": (
                properties.get_reference_density() if properties else None
            ),
        }


class SurfaceTensionForce:
    """表面張力項クラス"""

    def __init__(self, surface_tension: float = 0.07):
        """表面張力項を初期化

        Args:
            surface_tension: 表面張力係数（N/m）
        """
        self._name = "SurfaceTension"
        self._enabled = True
        self.surface_tension = surface_tension

    @property
    def name(self) -> str:
        """外力の名前を取得"""
        return self._name

    @property
    def enabled(self) -> bool:
        """外力が有効かどうかを取得"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """外力の有効/無効を設定"""
        self._enabled = value

    def compute(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> List[np.ndarray]:
        """表面張力項の寄与を計算"""
        if not self.enabled or levelset is None:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 表面張力係数の取得
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

        # 表面力の計算（符号付き距離関数の勾配方向）
        result = [np.zeros_like(v.data) for v in velocity.components]

        # 界面の法線方向ベクトルを計算
        grad_norm = 1e-10  # 分母ゼロ防止
        for i in range(velocity.ndim):
            grad_phi = levelset.gradient(i)
            grad_norm += grad_phi**2
        grad_norm = np.sqrt(grad_norm)

        # 表面力を法線方向に分配
        force = sigma * kappa * delta
        for i in range(velocity.ndim):
            grad_phi = levelset.gradient(i)
            result[i] = force * grad_phi / grad_norm

        return result

    def get_diagnostics(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> Dict[str, Any]:
        """診断情報を取得"""
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


class ForceTerm(TermBase):
    """外力項クラス"""

    def __init__(
        self,
        forces: Optional[List[ForceComponent]] = None,
        name: str = "Force",
        enabled: bool = True,
        logger=None,
    ):
        """外力項を初期化"""
        super().__init__(name=name, enabled=enabled, logger=logger)
        self.forces = forces or []

    def compute(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
        **kwargs,
    ) -> List[np.ndarray]:
        """外力項の寄与を計算"""
        if not self.enabled or not self.forces:
            return [np.zeros_like(v.data) for v in velocity.components]

        # 各外力の寄与を合計
        result = [np.zeros_like(v.data) for v in velocity.components]
        self._forces_contributions = []  # 診断用に各力の寄与を保存

        for force in self.forces:
            if force.enabled:
                force_contribution = force.compute(
                    velocity, levelset, properties, **kwargs
                )
                self._forces_contributions.append(
                    {
                        "name": force.name,
                        "contribution": force_contribution,
                    }
                )
                for i in range(len(result)):
                    result[i] += force_contribution[i]

        # 診断情報の更新
        self._update_force_diagnostics(velocity, levelset, properties)

        return result

    def _update_force_diagnostics(
        self,
        velocity: VectorField,
        levelset: LevelSetField,
        properties: PropertiesManager,
    ):
        """外力の診断情報を更新"""
        # 総力の大きさを計算
        total_force = 0.0
        force_ratios = {}

        for force_info in self._forces_contributions:
            force_mag = sum(np.sum(np.abs(f)) for f in force_info["contribution"])
            total_force += force_mag
            force_ratios[force_info["name"]] = force_mag

        # 相対的な寄与を計算
        if total_force > 0:
            for name, mag in force_ratios.items():
                force_ratios[name] = float(mag / total_force)

        # 診断情報を更新
        self._diagnostics.update(
            {
                "total_force": float(total_force),
                "force_ratios": force_ratios,
                "enabled_forces": len([f for f in self.forces if f.enabled]),
            }
        )

    def get_diagnostics(self) -> Dict[str, Any]:
        """項の診断情報を取得"""
        diag = super().get_diagnostics()

        # 各力の個別の診断情報を収集
        forces_diag = {}
        for force in self.forces:
            if force.enabled:
                forces_diag[force.name] = {
                    "ratio": self._diagnostics.get("force_ratios", {}).get(
                        force.name, 0.0
                    ),
                }

        diag.update(
            {
                "forces": forces_diag,
                "total_force": self._diagnostics.get("total_force", 0.0),
                "enabled_forces": self._diagnostics.get("enabled_forces", 0),
            }
        )

        return diag

    def add_force(self, force: ForceComponent):
        """外力を追加"""
        self.forces.append(force)

    def remove_force(self, force_name: str):
        """外力を削除"""
        self.forces = [f for f in self.forces if f.name != force_name]

    def get_force(self, force_name: str) -> Optional[ForceComponent]:
        """指定された名前の外力を取得"""
        for force in self.forces:
            if force.name == force_name:
                return force
        return None
