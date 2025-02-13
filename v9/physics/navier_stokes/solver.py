"""
二相流のためのNavier-Stokes方程式ソルバー

Navier-Stokes方程式の理論的導出と数値解法の詳細

1. 連続の方程式（質量保存則）
   微分形: ∂ρ/∂t + ∇⋅(ρu) = 0

2. 運動量保存則（基本形）
   ∂(ρu)/∂t + ∇⋅(ρuu) = -∇p + ∇⋅(μ(∇u + ∇uT)) + f

方程式の詳細な数学的展開:

I. 連続の方程式の展開
   1) ρの実質微分: Dρ/Dt = ∂ρ/∂t + u⋅∇ρ
   2) 連続の方程式の変形:
      ∂ρ/∂t + u⋅∇ρ + ρ∇⋅u = 0

II. 運動量保存則の詳細な導出
    1) 初期形: ∂(ρu)/∂t + ∇⋅(ρuu) = -∇p + ∇⋅(μ(∇u + ∇uT)) + f

    2) 各項の展開:
       a) 左辺第1項: ∂(ρu)/∂t
          = ρ∂u/∂t + u∂ρ/∂t

       b) 左辺第2項: ∇⋅(ρuu)
          = ρ(u⋅∇)u + u(u⋅∇ρ)

    3) 最終的な方程式形:
       ρ∂u/∂t + ρ(u⋅∇)u + u(u⋅∇ρ) = -∇p + ∇⋅(μ(∇u + ∇uT)) + f

    4) 速度の時間微分の形に変形:
       ∂u/∂t = -1/ρ (u⋅∇)u - 1/ρ u(u⋅∇ρ) - 1/ρ ∇p + 1/ρ ∇⋅(μ(∇u + ∇uT)) + 1/ρ f

数値解法の特徴:
1. 各項の物理的意味
   - 移流項 (-1/ρ (u⋅∇)u): 速度場の非線形な自己移流
   - 密度勾配項 (-1/ρ u(u⋅∇ρ)): 密度の空間変化による加速
   - 圧力項 (-1/ρ ∇p): 圧力勾配による運動
   - 粘性項 (1/ρ ∇⋅(μ(∇u + ∇uT))): 粘性による運動量拡散
   - 外力項 (1/ρ f): 重力や他の外部力

2. 数値スキームの仮定
   - 非圧縮性流体
   - 層流regime
   - 密度と粘性の空間的変化を許容
   - 外部力の一般的な取り扱い

実装上の注意:
- 高レイノルズ数流れでは追加のモデリングが必要
- 乱流効果は陽には考慮されていない
- 数値的安定性のため、各項の慎重な離散化が必要

数値的課題:
1. 移流項の非線形性
2. 密度と粘性の不連続性
3. 圧力-速度の結合
4. 界面での物理量の扱い
"""

from typing import Optional
import numpy as np

from core.field import VectorField, ScalarField
from .terms import (
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
    AccelerationTerm,
)


class NavierStokesSolver:
    """
    Navier-Stokes方程式のソルバー

    速度場、密度場、粘性場、圧力場から速度の時間微分を計算します。
    """

    def __init__(self):
        """ソルバーを初期化"""
        # 各項の初期化
        self.advection_term = AdvectionTerm()
        self.diffusion_term = DiffusionTerm()
        self.pressure_term = PressureTerm()
        self.acceleration_term = AccelerationTerm()

    def compute(
        self,
        velocity: VectorField,
        density: ScalarField,
        viscosity: ScalarField,
        pressure: ScalarField,
        force: Optional[VectorField] = None,
        **kwargs,
    ) -> VectorField:
        """
        速度の時間微分を計算

        式: ∂u/∂t = -u⋅∇u - 1/ρ u(u⋅∇ρ) - 1/ρ ∇p + 1/ρ ∇⋅(μ(∇u+∇uT)) + 1/ρ f

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            pressure: 圧力場
            force: 外力場（オプション）
            **kwargs: 追加のパラメータ

        Returns:
            速度の時間微分をVectorFieldとして返す
        """
        # 結果を格納するVectorFieldを作成
        result = VectorField(velocity.shape, velocity.dx)

        # 1. 移流項: -u⋅∇u
        advection = self.advection_term.compute(velocity)

        # 2. 密度勾配による加速度項: -1/ρ u(u⋅∇ρ)
        density_gradient = self.acceleration_term.compute(velocity, density)

        # 3. 粘性項: 1/ρ ∇⋅(μ(∇u+∇uT))
        diffusion = self.diffusion_term.compute(velocity, viscosity)

        # 4. 圧力項: -1/ρ ∇p
        pressure_grad = self.pressure_term.compute(velocity, pressure, density)

        # 5. 外力項: 1/ρ f
        if force is None:
            force = VectorField(velocity.shape, velocity.dx)

        # 各成分の時間微分を計算
        for i in range(velocity.ndim):
            result.components[i].data = (
                -advection.components[i].data  # 移流項
                + density_gradient.components[i].data  # 密度勾配項
                + diffusion.components[i].data  # 粘性項
                - pressure_grad.components[i].data  # 圧力項
                + force.components[i].data / np.maximum(density.data, 1e-10)  # 外力項
            )

        # 診断情報の更新
        self._update_diagnostics(
            advection=advection,
            density_gradient=density_gradient,
            diffusion=diffusion,
            pressure_grad=pressure_grad,
            force=force,
            result=result,
        )

        return result

    def _update_diagnostics(self, **fields: VectorField) -> None:
        """
        診断情報を更新

        Args:
            **fields: 更新に使用する各VectorField
        """
        self._diagnostics = {}
        for name, field in fields.items():
            max_value = max(np.max(np.abs(comp.data)) for comp in field.components)
