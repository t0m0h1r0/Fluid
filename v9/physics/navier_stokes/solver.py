"""Navier-Stokesソルバーを提供するモジュール

このモジュールは、非圧縮性Navier-Stokes方程式を解くためのソルバーを実装します。
分離解法（Projection Method）を用いて、各時間ステップで予測子-修正子法を適用します。
"""

from typing import Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass
from core.solver import TemporalSolver
from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from .terms.advection import AdvectionTerm
from .terms.diffusion import DiffusionTerm
from .terms.pressure import PressureTerm
from .terms.force import ForceTerm, GravityForce, SurfaceTensionForce

@dataclass
class NavierStokesParameters:
    """Navier-Stokesソルバーのパラメータ
    
    Attributes:
        pressure_iterations: 圧力Poisson方程式の最大反復回数
        pressure_tolerance: 圧力Poisson方程式の収束判定閾値
        divergence_tolerance: 非圧縮性条件の許容誤差
        cfl_safety_factor: CFL条件の安全係数
    """
    pressure_iterations: int = 100
    pressure_tolerance: float = 1e-6
    divergence_tolerance: float = 1e-5
    cfl_safety_factor: float = 0.5

class NavierStokesSolver(TemporalSolver):
    """Navier-Stokesソルバークラス
    
    非圧縮性Navier-Stokes方程式を解くソルバーを実装します。
    分離解法を用いて、圧力と速度を分離して解きます。
    """
    
    def __init__(self,
                 params: Optional[NavierStokesParameters] = None,
                 use_weno: bool = True,
                 poisson_solver: Optional['PoissonSolver'] = None):
        """Navier-Stokesソルバーを初期化
        
        Args:
            params: ソルバーのパラメータ
            use_weno: WENOスキームを使用するかどうか
            poisson_solver: 圧力Poisson方程式のソルバー
        """
        super().__init__(name="NavierStokes")
        self.params = params or NavierStokesParameters()
        
        # 各項の初期化
        self.advection = AdvectionTerm(use_weno=use_weno)
        self.diffusion = DiffusionTerm()
        self.pressure = PressureTerm()
        self.force = ForceTerm()
        
        # デフォルトの外力として重力を追加
        self.force.add_force(GravityForce())
        
        # Poissonソルバーの設定
        if poisson_solver is None:
            from physics.poisson.sor import SORSolver
            self.poisson_solver = SORSolver(
                omega=1.5,  # デフォルトの緩和係数
                tolerance=self.params.pressure_tolerance,
                max_iterations=self.params.pressure_iterations
            )
        else:
            self.poisson_solver = poisson_solver
        
        # 診断用の変数
        self._max_divergence = 0.0
        self._pressure_iterations = 0
        
    def initialize(self, velocity: VectorField,
                  levelset: Optional[LevelSetField] = None,
                  **kwargs):
        """ソルバーの初期化
        
        Args:
            velocity: 初期速度場
            levelset: Level Set場（二相流体の場合）
            **kwargs: 追加のパラメータ
        """
        # 表面張力の初期化（二相流体の場合）
        if levelset is not None and not any(
            isinstance(f, SurfaceTensionForce) for f in self.force.forces
        ):
            self.force.add_force(SurfaceTensionForce())
        
        # 各項の初期化
        self.advection.initialize(**kwargs)
        self.diffusion.initialize(**kwargs)
        self.pressure.initialize(**kwargs)
        self.force.initialize(**kwargs)
    
    def compute_timestep(self, velocity: VectorField, **kwargs) -> float:
        """CFL条件に基づく時間刻み幅を計算
        
        各項からの制限を考慮して、最小の時間刻み幅を選択します。
        
        Args:
            velocity: 現在の速度場
            **kwargs: 追加のパラメータ
            
        Returns:
            計算された時間刻み幅
        """
        # 各項からの制限を取得
        dt_advection = self.advection.compute_timestep(velocity, **kwargs)
        dt_diffusion = self.diffusion.compute_timestep(velocity, **kwargs)
        
        # 最も厳しい制限を採用
        dt = min(dt_advection, dt_diffusion)
        
        # 安全係数を適用
        return self.params.cfl_safety_factor * dt
    
    def advance(self, dt: float, velocity: VectorField,
                pressure: ScalarField,
                levelset: Optional[LevelSetField] = None,
                **kwargs) -> Dict[str, Any]:
        """1時間ステップを進める
        
        分離解法を用いて、以下のステップで解きます：
        1. 予測子ステップ：圧力項を除いた方程式を解く
        2. 圧力補正ステップ：圧力Poisson方程式を解く
        3. 修正子ステップ：速度場を補正
        
        Args:
            dt: 時間刻み幅
            velocity: 現在の速度場
            pressure: 圧力場
            levelset: Level Set場（二相流体の場合）
            **kwargs: 追加のパラメータ
            
        Returns:
            計算結果と統計情報を含む辞書
        """
        # 予測子ステップ
        velocity_star = self._predictor_step(dt, velocity, levelset, **kwargs)
        
        # 圧力補正ステップ
        pressure_correction = self._pressure_correction_step(
            dt, velocity_star, pressure, **kwargs
        )
        
        # 修正子ステップ
        velocity_new = self._corrector_step(
            dt, velocity_star, pressure_correction, **kwargs
        )
        
        # 非圧縮性条件のチェック
        div = velocity_new.divergence()
        self._max_divergence = np.max(np.abs(div))
        
        # 圧力場の更新
        pressure.data += pressure_correction.data
        
        # 統計情報の収集
        stats = self._collect_diagnostics(
            velocity_new, pressure, levelset, div, **kwargs
        )
        
        return {
            'velocity': velocity_new,
            'pressure': pressure,
            'converged': self._max_divergence < self.params.divergence_tolerance,
            'stats': stats
        }
    
    def _predictor_step(self, dt: float, velocity: VectorField,
                       levelset: Optional[LevelSetField] = None,
                       **kwargs) -> VectorField:
        """予測子ステップ
        
        圧力項を除いた方程式を解きます。
        """
        # 各項の寄与を計算
        advection = self.advection.compute(velocity, **kwargs)
        diffusion = self.diffusion.compute(velocity, **kwargs)
        force = self.force.compute(velocity, levelset=levelset, **kwargs)
        
        # 予測速度場を計算
        result = VectorField(velocity.shape, velocity.dx)
        for i in range(velocity.ndim):
            result.components[i].data = (
                velocity.components[i].data +
                dt * (advection[i] + diffusion[i] + force[i])
            )
        
        return result
    
    def _pressure_correction_step(self, dt: float,
                                velocity: VectorField,
                                pressure: ScalarField,
                                **kwargs) -> ScalarField:
        """圧力補正ステップ
        
        圧力Poisson方程式を解きます。
        
        Args:
            dt: 時間刻み幅
            velocity: 予測速度場
            pressure: 現在の圧力場
            **kwargs: 追加のパラメータ（密度など）
            
        Returns:
            圧力補正場
        """
        # 発散を計算
        div = velocity.divergence()
        
        # 圧力Poisson方程式の右辺
        # 非圧縮条件: ∇・u^(n+1) = 0 より
        # ∇²p = ρ/dt * ∇・u^*
        density = kwargs.get('density', None)
        if density is not None:
            rhs = density.data * div / dt
        else:
            rhs = div / dt
        
        # 境界条件の設定
        # 圧力の境界条件は速度場の境界条件から導出
        self.poisson_solver.boundary_conditions = self._get_pressure_boundary_conditions()
        
        # 圧力補正値の初期推定値
        p_corr = ScalarField(pressure.shape, pressure.dx)
        
        # Poisson方程式を解く
        try:
            p_corr.data = self.poisson_solver.solve(
                initial_solution=np.zeros_like(pressure.data),
                rhs=rhs,
                dx=velocity.dx
            )
            self._pressure_iterations = self.poisson_solver.iteration_count
            
        except RuntimeError as e:
            # ソルバーが収束しない場合の処理
            self.logger.warning(f"圧力ソルバーが収束しませんでした: {str(e)}")
            self._pressure_iterations = self.params.pressure_iterations
        
        return p_corr
    
    def _corrector_step(self, dt: float,
                       velocity: VectorField,
                       pressure_correction: ScalarField,
                       **kwargs) -> VectorField:
        """修正子ステップ
        
        圧力勾配に基づいて速度場を補正します。
        """
        return self.pressure.compute_correction(
            velocity, pressure_correction, dt, **kwargs
        )
    
    def _collect_diagnostics(self, velocity: VectorField,
                           pressure: ScalarField,
                           levelset: Optional[LevelSetField],
                           divergence: np.ndarray,
                           **kwargs) -> Dict[str, Any]:
        """診断情報の収集"""
        diag = {
            'max_velocity': max(np.max(np.abs(v.data)) 
                              for v in velocity.components),
            'max_pressure': np.max(np.abs(pressure.data)),
            'max_divergence': self._max_divergence,
            'pressure_iterations': self._pressure_iterations,
            'kinetic_energy': 0.5 * sum(
                np.sum(v.data**2) for v in velocity.components
            ) * velocity.dx**velocity.ndim
        }
        
        # 各項の診断情報
        diag.update({
            'advection': self.advection.get_diagnostics(velocity, **kwargs),
            'diffusion': self.diffusion.get_diagnostics(velocity, **kwargs),
            'pressure': self.pressure.get_diagnostics(velocity, pressure, **kwargs),
            'force': self.force.get_diagnostics(velocity, levelset=levelset, **kwargs)
        })
        
        return diag