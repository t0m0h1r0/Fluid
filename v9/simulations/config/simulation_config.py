"""シミュレーション設定を管理するモジュール

このモジュールは、二相流シミュレーションの設定を管理します。
YAMLファイルから設定を読み込み、適切なデータ構造に変換します。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml

from physics.properties import FluidProperties


@dataclass
class DomainConfig:
    """計算領域の設定"""

    dimensions: List[int]
    size: List[float]

    def validate(self):
        """設定値の妥当性を検証"""
        if len(self.dimensions) != 3 or len(self.size) != 3:
            raise ValueError("dimensionsとsizeは3次元である必要があります")
        if any(n <= 0 for n in self.dimensions):
            raise ValueError("グリッド数は正の整数である必要があります")
        if any(s <= 0 for s in self.size):
            raise ValueError("領域サイズは正の値である必要があります")


@dataclass
class PhysicsConfig:
    """物理パラメータの設定"""

    gravity: float = 9.81
    surface_tension: float = 0.072


@dataclass
class PhaseConfig:
    """各相の物性値設定"""

    density: float
    viscosity: float
    surface_tension: Optional[float] = None

    def to_properties(self) -> FluidProperties:
        """FluidPropertiesインスタンスに変換"""
        return FluidProperties(
            density=self.density,
            viscosity=self.viscosity,
            surface_tension=self.surface_tension,
        )


@dataclass
class SolverConfig:
    """ソルバーの設定"""

    time_integrator: str = "rk4"
    use_weno: bool = True
    weno_order: int = 5
    level_set: Dict[str, Any] = field(
        default_factory=lambda: {
            "epsilon": 1.0e-2,
            "reinit_interval": 5,
            "reinit_steps": 2,
        }
    )
    pressure_solver: Dict[str, Any] = field(
        default_factory=lambda: {
            "omega": 1.5,
            "tolerance": 1e-6,
            "max_iterations": 1000,
        }
    )


@dataclass
class TimeConfig:
    """時間発展の設定"""

    max_time: float
    cfl: float = 0.5
    min_dt: float = 1e-6
    max_dt: float = 1.0
    save_interval: float = 0.1


@dataclass
class ObjectConfig:
    """物体の設定"""

    type: str
    phase: str
    center: List[float]
    radius: float


@dataclass
class InitialConditionConfig:
    """初期条件の設定"""

    background_layer: Optional[float] = None
    objects: List[ObjectConfig] = field(default_factory=list)
    velocity: Dict[str, Any] = field(default_factory=lambda: {"type": "zero"})


@dataclass
class OutputConfig:
    """出力の設定"""

    directory: str = "results"
    format: str = "vti"
    variables: List[str] = field(
        default_factory=lambda: ["velocity", "pressure", "levelset"]
    )


@dataclass
class SimulationConfig:
    """シミュレーション全体の設定"""

    domain: DomainConfig
    physics: PhysicsConfig
    phases: Dict[str, PhaseConfig]
    solver: SolverConfig
    time: TimeConfig
    initial_condition: InitialConditionConfig
    output: OutputConfig

    @classmethod
    def from_yaml(cls, filepath: str) -> "SimulationConfig":
        """YAMLファイルから設定を読み込む

        Args:
            filepath: 設定ファイルのパス

        Returns:
            読み込まれた設定
        """
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 各セクションの設定を変換
        domain = DomainConfig(**config_dict["domain"])
        physics = PhysicsConfig(**config_dict.get("physics", {}))
        phases = {
            name: PhaseConfig(**props)
            for name, props in config_dict.get("phases", {}).items()
        }
        solver = SolverConfig(**config_dict.get("solver", {}))
        time = TimeConfig(**config_dict.get("time", {}))

        # 初期条件の設定を変換
        ic_dict = config_dict.get("initial_condition", {})
        objects = [ObjectConfig(**obj) for obj in ic_dict.get("objects", [])]
        initial_condition = InitialConditionConfig(
            background_layer=ic_dict.get("background_layer"),
            objects=objects,
            velocity=ic_dict.get("velocity", {"type": "zero"}),
        )

        output = OutputConfig(**config_dict.get("output", {}))

        return cls(
            domain=domain,
            physics=physics,
            phases=phases,
            solver=solver,
            time=time,
            initial_condition=initial_condition,
            output=output,
        )

    def save(self, filepath: str):
        """設定をYAMLファイルに保存

        Args:
            filepath: 保存先のパス
        """
        # dataclassをdictに変換
        config_dict = {
            "domain": self.domain.__dict__,
            "physics": self.physics.__dict__,
            "phases": {name: phase.__dict__ for name, phase in self.phases.items()},
            "solver": self.solver.__dict__,
            "time": self.time.__dict__,
            "initial_condition": {
                "background_layer": self.initial_condition.background_layer,
                "objects": [obj.__dict__ for obj in self.initial_condition.objects],
                "velocity": self.initial_condition.velocity,
            },
            "output": self.output.__dict__,
        }

        # YAMLファイルとして保存
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
