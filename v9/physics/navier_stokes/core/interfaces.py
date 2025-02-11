from typing import Protocol, Dict, Any, List, Optional, TypeVar, Generic, Tuple
import numpy as np

from core.field import VectorField

StateType = TypeVar("StateType")


class NavierStokesTerm(Protocol):
    """Navier-Stokes方程式の項のプロトコル"""

    @property
    def name(self) -> str: ...

    @property
    def enabled(self) -> bool: ...

    def compute(self, velocity: VectorField, **kwargs) -> List[np.ndarray]: ...

    def compute_timestep(self, velocity: VectorField, **kwargs) -> float: ...

    def get_diagnostics(self) -> Dict[str, Any]: ...


class NavierStokesSolver(Protocol, Generic[StateType]):
    """Navier-Stokesソルバーのプロトコル"""

    @property
    def time(self) -> float: ...

    def initialize(self, state: Optional[StateType] = None) -> None: ...

    def step_forward(self, dt: Optional[float] = None, **kwargs) -> Dict[str, Any]: ...

    def get_state(self) -> Tuple[StateType, Dict[str, Any]]: ...
