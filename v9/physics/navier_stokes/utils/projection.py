"""圧力投影法を提供するモジュール

このモジュールは、非圧縮性Navier-Stokes方程式のための圧力投影法を実装します。
古典的な投影法と回転形式の投影法の両方を提供します。
"""

from typing import Optional, Tuple
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager
from physics.poisson import PoissonSolver
from ..core.interfaces import PressureProjection


class ClassicProjection(PressureProjection):
    """古典的な圧力投影法

    Chorin (1968) による古典的な投影法を実装します。
    投影によって離散的な非圧縮性を厳密に満たします。
    """

    def __init__(self, poisson_solver: PoissonSolver):
        """古典的な投影法を初期化

        Args:
            poisson_solver: 圧力ポアソンソルバー
        """
        self.poisson_solver = poisson_solver

    def project(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        dt: float,
        levelset: Optional[LevelSetField] = None,
        properties: Optional[PropertiesManager] = None,
    ) -> Tuple[VectorField, ScalarField]:
        """速度場を非圧縮に投影

        Args:
            velocity: 速度場
            pressure: 圧力場
            dt: 時間刻み幅
            levelset: レベルセット場（オプション）
            properties: 物性値マネージャー（オプション）

        Returns:
            (投影された速度場, 更新された圧力場)のタプル
        """
        # 密度場の取得
        density = (
            properties.get_density(levelset).data
            if properties and levelset
            else np.ones_like(velocity.components[0].data)
        )

        # 発散の計算
        div = velocity.divergence()

        # ポアソン方程式の右辺を設定
        rhs = -div / dt

        # 圧力補正値の計算
        p_corr = ScalarField(velocity.shape, velocity.dx)
        p_corr.data = self.poisson_solver.solve(
            initial_solution=np.zeros_like(pressure.data),
            rhs=rhs,
            dx=velocity.dx,
        )

        # 速度場の補正
        velocity_new = velocity.copy()
        for i, component in enumerate(velocity_new.components):
            grad_p = np.gradient(p_corr.data, velocity.dx, axis=i)
            component.data -= dt * grad_p / density

        # 圧力場の更新
        pressure_new = pressure.copy()
        pressure_new.data += p_corr.data

        return velocity_new, pressure_new


class RotationalProjection(PressureProjection):
    """回転形式の圧力投影法

    Timmermans et al. (1996) による回転形式の投影法を実装します。
    より高精度な圧力場の計算が可能です。
    """

    def __init__(self, poisson_solver: PoissonSolver, include_viscous: bool = True):
        """回転形式の投影法を初期化

        Args:
            poisson_solver: 圧力ポアソンソルバー
            include_viscous: 粘性項を含めるかどうか
        """
        self.poisson_solver = poisson_solver
        self.include_viscous = include_viscous

    def project(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        dt: float,
        levelset: Optional[LevelSetField] = None,
        properties: Optional[PropertiesManager] = None,
    ) -> Tuple[VectorField, ScalarField]:
        """速度場を非圧縮に投影

        Args:
            velocity: 速度場
            pressure: 圧力場
            dt: 時間刻み幅
            levelset: レベルセット場（オプション）
            properties: 物性値マネージャー（オプション）

        Returns:
            (投影された速度場, 更新された圧力場)のタプル
        """
        # 密度と粘性係数の取得
        if properties and levelset:
            density = properties.get_density(levelset).data
            viscosity = (
                properties.get_viscosity(levelset).data
                if self.include_viscous
                else None
            )
        else:
            density = np.ones_like(velocity.components[0].data)
            viscosity = None

        # 渦度の計算
        curl = velocity.curl()

        # 圧力を含む運動量の発散を計算
        div_momentum = velocity.divergence()
        if viscosity is not None:
            # 粘性項の寄与を追加
            for i, v_i in enumerate(velocity.components):
                momentum = density * v_i.data
                div_momentum += sum(
                    np.gradient(
                        viscosity * np.gradient(momentum, velocity.dx, axis=j),
                        velocity.dx,
                        axis=j,
                    )
                    for j in range(velocity.ndim)
                )

        # ポアソン方程式の右辺を設定
        rhs = -div_momentum / dt

        # 圧力補正値の計算
        p_corr = ScalarField(velocity.shape, velocity.dx)
        p_corr.data = self.poisson_solver.solve(
            initial_solution=np.zeros_like(pressure.data),
            rhs=rhs,
            dx=velocity.dx,
        )

        # 速度場の補正（渦度を保存）
        velocity_new = velocity.copy()
        for i, component in enumerate(velocity_new.components):
            # 圧力勾配による補正
            grad_p = np.gradient(p_corr.data, velocity.dx, axis=i)
            component.data -= dt * grad_p / density

            if viscosity is not None:
                # 粘性による補正
                component.data += (
                    dt
                    * sum(
                        np.gradient(
                            viscosity
                            * np.gradient(component.data, velocity.dx, axis=j),
                            velocity.dx,
                            axis=j,
                        )
                        for j in range(velocity.ndim)
                    )
                    / density
                )

        # 圧力場の更新
        pressure_new = pressure.copy()
        pressure_new.data += p_corr.data

        # 非圧縮性の確認
        div_final = velocity_new.divergence()
        if np.max(np.abs(div_final)) > 1e-10:
            # 追加の補正が必要な場合は再投影
            velocity_new, pressure_new = self.project(
                velocity_new, pressure_new, dt, levelset, properties
            )

        return velocity_new, pressure_new
