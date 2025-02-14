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

from typing import Optional, Dict, Any

from core.field import VectorField, ScalarField
from .terms import (
    AdvectionTerm,
    DiffusionTerm,
    PressureTerm,
    AccelerationTerm,
)


class NavierStokesSolver:
    """Navier-Stokes方程式のソルバー（改良版）"""

    def __init__(self):
        """ソルバーを初期化"""
        self.advection_term = AdvectionTerm()
        self.diffusion_term = DiffusionTerm()
        self.pressure_term = PressureTerm()
        self.acceleration_term = AccelerationTerm()
        self._diagnostics: Dict[str, Any] = {}

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
        速度の時間微分を計算（新しい演算子を活用）

        Args:
            velocity: 速度場
            density: 密度場
            viscosity: 粘性場
            pressure: 圧力場
            force: 外力場（オプション）

        Returns:
            速度の時間微分をVectorFieldとして返す
        """
        # デフォルトの外力場を初期化
        if force is None:
            force = VectorField(velocity.shape, velocity.dx)

        # 各項を新しい演算子を使用して計算
        advection = self.advection_term.compute(velocity)
        density_gradient = self.acceleration_term.compute(velocity, density)
        diffusion = self.diffusion_term.compute(velocity, viscosity)
        pressure_grad = self.pressure_term.compute(velocity, pressure, density)

        # 密度の逆数を計算（ゼロ除算を防止）
        inv_density = 1.0 / (density + ScalarField(density.shape, density.dx, 1e-10))

        # 時間微分の計算（新しい演算子を活用）
        result = (
            -advection  # 移流項
            + density_gradient  # 密度勾配項
            + diffusion  # 粘性項
            - pressure_grad  # 圧力項
            + force * inv_density  # 外力項
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

    def _update_diagnostics(self, **fields: VectorField):
        """診断情報を更新（新しいメソッドを活用）"""
        self._diagnostics = {
            name: {
                "max_magnitude": float(field.magnitude().max()),
                "min_magnitude": float(field.magnitude().min()),
                "mean_magnitude": float(field.magnitude().mean()),
                "component_norms": [float(comp.norm()) for comp in field.components],
            }
            for name, field in fields.items()
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        return self._diagnostics.copy()
