"""二相流シミュレーションの主要クラスを提供するモジュール

リファクタリングされたphysics/パッケージに対応した更新版
"""

from typing import Dict, Any, Tuple, Optional

from physics.levelset import LevelSetPropertiesManager, LevelSetSolver
from physics.navier_stokes import solvers, terms
from physics.time_evolution import TimeEvolutionSolver


from .config import SimulationConfig
from .state import SimulationState
from .initializer import SimulationInitializer


class TwoPhaseFlowSimulator:
    """二相流シミュレーションクラス

    リファクタリングされたphysics/パッケージを活用した
    高度な二相流シミュレーションを実現
    """

    def __init__(
        self, 
        config: SimulationConfig, 
        logger=None
    ):
        """シミュレーションを初期化

        Args:
            config: シミュレーション設定
            logger: ロガー（オプション）
        """
        self.config = config
        self.logger = logger

        # 物性値マネージャーの初期化
        self._properties = LevelSetPropertiesManager(
            phase1=config.phases["water"].to_properties(),
            phase2=config.phases["nitrogen"].to_properties(),
        )

        # ソルバーコンポーネントの初期化
        self._initialize_solvers()

    def _initialize_solvers(self):
        """ソルバーコンポーネントを初期化"""
        # 移流項
        advection_term = terms.AdvectionTerm(
            use_weno=self.config.solver.use_weno,
            weno_order=self.config.solver.weno_order
        )

        # 粘性項
        diffusion_term = terms.DiffusionTerm(
            use_conservative=True
        )

        # 外力項（重力など）
        gravity_force = terms.GravityForce(
            gravity=self.config.physics.gravity
        )
        force_term = terms.ForceTerm(forces=[gravity_force])

        # 圧力項
        pressure_term = terms.PressureTerm()

        # Navier-Stokesソルバーの設定
        self._ns_solver = solvers.ProjectionSolver(
            terms=[
                advection_term, 
                diffusion_term, 
                force_term, 
                pressure_term
            ],
            properties=self._properties,
            use_rotational=True
        )

        # Level Setソルバーの設定
        self._ls_solver = LevelSetSolver(
            use_weno=self.config.solver.use_weno,
            weno_order=self.config.solver.weno_order
        )

        # 時間発展ソルバー
        self._time_solver = TimeEvolutionSolver(
            terms=[self._ns_solver, self._ls_solver],
            integrator_type=self.config.solver.time_integrator
        )

    def initialize(self, state: Optional[SimulationState] = None):
        """シミュレーションを初期化

        Args:
            state: 初期状態（オプション）
        """
        if state is None:
            initializer = SimulationInitializer(self.config)
            state = initializer.create_initial_state()

        self._time_solver.initialize(state)

    def step_forward(self) -> Tuple[SimulationState, Dict[str, Any]]:
        """1時間ステップを進める

        Returns:
            更新された状態と診断情報
        """
        return self._time_solver.step_forward()

    def save_checkpoint(self, filepath: str):
        """チェックポイントを保存

        Args:
            filepath: 保存先パス
        """
        self._time_solver.save_checkpoint(filepath)

    def load_checkpoint(self, filepath: str):
        """チェックポイントから復元

        Args:
            filepath: チェックポイントファイルのパス
        """
        checkpoint_state = self._time_solver.load_checkpoint(filepath)
        self.initialize(checkpoint_state)