from physics.levelset import LevelSetMethod
from physics.navier_stokes.core import NavierStokesSolver
from numerics.time_evolution.euler import ForwardEuler as TimeIntegrator
from .config import SimulationConfig
from .state import SimulationState
from .initializer import SimulationInitializer


class TwoPhaseFlowSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.time_integrator = TimeIntegrator()
        self.levelset_method = LevelSetMethod()
        self.navier_stokes_solver = NavierStokesSolver()

    def initialize(self):
        initializer = SimulationInitializer(self.config)
        self.state = initializer.create_initial_state()

    def run(self, max_iterations):
        for _ in range(max_iterations):
            self.state = self._step_forward()
            self._output_state()

    def _step_forward(self):
        levelset = self.state.levelset
        velocity = self.state.velocity

        density = self.levelset_method.to_density(levelset)
        viscosity = self.levelset_method.to_viscosity(levelset)

        # 速度場を引数として渡す
        levelset_deriv = self.levelset_method.run(levelset, velocity)

        pressure, velocity_deriv = self.navier_stokes_solver.run(
            velocity=velocity,
            density=density,
            viscosity=viscosity,
        )

        dt = self._compute_dt()

        new_levelset = self.time_integrator.run(
            data=levelset.data, derivative=levelset_deriv, dt=dt
        )
        new_velocity = self.time_integrator.run(
            data=velocity, derivative=velocity_deriv, dt=dt
        )

        return SimulationState(
            time=self.state.time + dt,
            velocity=new_velocity,
            levelset=new_levelset,
            pressure=pressure,
        )

    def _compute_dt(self):
        # 安定性条件に基づきdtを計算
        return 0.01

    def _output_state(self):
        # 現在の状態を出力
        pass
