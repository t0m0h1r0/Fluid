"""圧力投影法を提供するモジュール

このモジュールは、非圧縮性流れのための圧力投影法を実装します。
速度場の発散をゼロにするように圧力場を計算し、速度場を補正します。
"""

from typing import Tuple, Optional
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager
from physics.poisson import PoissonSolver


class ClassicProjection:
    """古典的な圧力投影法"""

    def __init__(self, poisson_solver: PoissonSolver, logger=None):
        """圧力投影法を初期化

        Args:
            poisson_solver: 圧力ポアソンソルバー
            logger: ロガー
        """
        self.poisson_solver = poisson_solver
        self.logger = logger

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
            velocity: 補正前の速度場
            pressure: 前ステップの圧力場
            dt: 時間刻み幅
            levelset: レベルセット場（オプション）
            properties: 物性値マネージャー（オプション）

        Returns:
            (補正された速度場, 更新された圧力場)のタプル
        """
        try:
            # 速度場の発散を計算
            div = velocity.divergence()

            # 圧力ポアソン方程式の右辺
            rhs = ScalarField(velocity.shape, velocity.dx)
            rhs.data = -div.data / dt

            # 圧力補正値の計算
            p_corr = ScalarField(velocity.shape, velocity.dx)
            p_corr.data = self.poisson_solver.solve(
                initial_solution=np.zeros_like(pressure.data),
                rhs=rhs.data,
                dx=velocity.dx,
            )

            # 密度を考慮した速度場の補正
            velocity_new = velocity.copy()
            density = (
                properties.get_density(levelset).data
                if properties and levelset
                else np.ones_like(pressure.data)
            )

            for i, component in enumerate(velocity_new.components):
                grad_p = np.gradient(p_corr.data, velocity.dx, axis=i)
                component.data -= dt * grad_p / density

            # 圧力場の更新
            pressure_new = pressure.copy()
            pressure_new.data += p_corr.data

            return velocity_new, pressure_new

        except Exception as e:
            if self.logger:
                self.logger.error(f"圧力投影中にエラー: {e}")
            raise
