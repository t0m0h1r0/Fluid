"""Navier-Stokes方程式の圧力項を提供するモジュール

このモジュールは、Navier-Stokes方程式の圧力項の計算を実装します。
圧力勾配項を保存形式で離散化し、境界条件との整合性を保ちます。
"""

import numpy as np
from typing import List, Dict, Any, Optional
from core.field import VectorField, ScalarField
from core.boundary import BoundaryCondition
from ..base_term import NavierStokesTerm


class PressureTerm(NavierStokesTerm):
    """圧力項クラス

    圧力項 -1/ρ ∇p を計算します。
    保存形式での離散化を行い、境界条件との整合性を保ちます。

    Attributes:
        boundary_conditions: 各方向の境界条件
        staggered: スタガード格子を使用するかどうか
    """

    def __init__(
        self,
        boundary_conditions: Optional[List[BoundaryCondition]] = None,
        staggered: bool = True,
    ):
        """圧力項を初期化

        Args:
            boundary_conditions: 各方向の境界条件
            staggered: スタガード格子を使用するかどうか
        """
        super().__init__(name="Pressure")
        self.boundary_conditions = boundary_conditions
        self.staggered = staggered

        # 圧力勾配の補間に使用する係数を初期化
        if staggered:
            self._init_interpolation_coeffs()

    def _init_interpolation_coeffs(self):
        """スタガード格子での補間係数を初期化"""
        # 2次精度中心差分の係数
        self.interp_coeffs = np.array([0.5, 0.5])

        # 4次精度補間の係数
        self.interp_coeffs_4th = np.array([-1 / 16, 9 / 16, 9 / 16, -1 / 16])

    def compute(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        density: Optional[ScalarField] = None,
        **kwargs
    ) -> List[np.ndarray]:
        """圧力項の寄与を計算

        Args:
            velocity: 現在の速度場
            pressure: 圧力場
            density: 密度場（Noneの場合は一様密度を仮定）
            **kwargs: 未使用

        Returns:
            各方向の速度成分への寄与のリスト
        """
        if not self.enabled:
            return [np.zeros_like(v.data) for v in velocity.components]

        dx = velocity.dx
        result = []

        # 密度場の設定
        if density is None:
            rho = np.ones_like(pressure.data)
        else:
            rho = density.data

        # 境界条件の適用
        p = pressure.data.copy()
        if self.boundary_conditions:
            for i, bc in enumerate(self.boundary_conditions):
                if bc is not None:
                    p = bc.apply_all(p, i)

        if self.staggered:
            # スタガード格子での計算
            for i in range(velocity.ndim):
                # 圧力勾配の計算（スタガード点で）
                grad_p = np.gradient(p, dx, axis=i)

                # 密度の補間（スタガード点へ）
                rho_stag = (np.roll(rho, -1, axis=i) + rho) * 0.5

                # 圧力項の計算
                result.append(-grad_p / rho_stag)
        else:
            # コロケート格子での計算
            for i in range(velocity.ndim):
                # 圧力勾配の計算
                grad_p = np.gradient(p, dx, axis=i)

                # 圧力項の計算
                result.append(-grad_p / rho)

        return result

    def compute_correction(
        self, 
        velocity: VectorField, 
        pressure: ScalarField, 
        dt: float,
        density: Optional[ScalarField] = None,
        **kwargs  # プロパティを許容するための可変キーワード引数
    ) -> VectorField:
        """速度場の圧力補正を計算

        Args:
            velocity: 補正する速度場
            pressure: 圧力場
            dt: 時間刻み幅
            density: 密度場（オプション）
            **kwargs: 追加のパラメータ（properties等）

        Returns:
            補正された速度場
        """
        # 各項の寄与を計算
        pressure_terms = self.compute(velocity, pressure, density)

        # 補正された速度場を作成
        corrected = VectorField(velocity.shape, velocity.dx)
        for i, (v, dp) in enumerate(zip(velocity.components, pressure_terms)):
            corrected.components[i].data = v.data + dt * dp

        return corrected

    def project_velocity(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        density: Optional[ScalarField] = None,
    ) -> VectorField:
        """速度場を非圧縮条件に射影

        Args:
            velocity: 射影する速度場
            pressure: 圧力場
            density: 密度場

        Returns:
            射影された速度場
        """
        # 発散を計算
        div = velocity.divergence()

        # 圧力場を補正
        p_corr = pressure.copy()
        p_corr.data = -div.data * velocity.dx**2

        # 速度場を補正
        return self.compute_correction(velocity, p_corr, 1.0, density)

    def get_diagnostics(
        self,
        velocity: VectorField,
        pressure: ScalarField,
        density: Optional[ScalarField] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """圧力項の診断情報を取得"""
        diag = super().get_diagnostics(
            velocity, pressure=pressure, density=density, **kwargs
        )

        # 圧力勾配の大きさを計算
        grad_p = np.array(
            [
                np.gradient(pressure.data, velocity.dx, axis=i)
                for i in range(velocity.ndim)
            ]
        )
        grad_p_mag = np.sqrt(np.sum(grad_p**2, axis=0))

        # 有効な値のみを使用
        valid_indices = np.isfinite(grad_p_mag)
        max_grad_p = np.max(grad_p_mag[valid_indices]) if np.any(valid_indices) else 0.0
        
        # 平面ではなく3D空間全体の最大値を取得
        diag.update(
            {
                "grid_type": "staggered" if self.staggered else "collocated",
                "max_pressure": np.nanmax(pressure.data),
                "max_pressure_gradient": max_grad_p,
                "pressure_l2norm": np.sqrt(
                    np.nansum(pressure.data**2) * velocity.dx**velocity.ndim
                ),
            }
        )
        return diag