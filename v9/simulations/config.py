"""シミュレーション設定を管理するモジュール

リファクタリングされたphysics/パッケージに対応した更新版
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import yaml

from physics.levelset import FluidPhaseProperties


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

    def to_properties(self) -> FluidPhaseProperties:
        """FluidPropertiesインスタンスに変換"""
        return FluidPhaseProperties(
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
            "method": "sor",
            "relaxation_parameter": 1.5,
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

    directory: str = "results/visualization"
    output_dir: str = "results/visualization"
    format: str = "png"
    dpi: int = 300
    colormap: str = "viridis"
    show_colorbar: bool = True
    show_axes: bool = True
    show_grid: bool = False

    fields: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "velocity": {"enabled": False, "plot_types": ["vector"]},
            "pressure": {"enabled": True, "plot_types": ["scalar"]},
            "levelset": {"enabled": True, "plot_types": ["interface"]},
        }
    )

    slices: Dict[str, List[Union[str, float]]] = field(
        default_factory=lambda: {"axes": ["xy", "xz", "yz"], "positions": [0.5]}
    )

    def __post_init__(self):
        """初期化後の処理"""
        if self.output_dir:
            self.directory = self.output_dir


@dataclass
class SimulationConfig:
    """シミュレーション全体の設定"""

    domain: DomainConfig
    physics: PhysicsConfig
    phases: Dict[str, PhaseConfig]
    solver: SolverConfig
    time: TimeConfig
    initial_condition: InitialConditionConfig
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def output_dir(self) -> str:
        """出力ディレクトリを取得"""
        return self.output.directory

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

        return cls(
            domain=DomainConfig(**config_dict.get("domain", {})),
            physics=PhysicsConfig(**config_dict.get("physics", {})),
            phases={
                name: PhaseConfig(**props)
                for name, props in config_dict.get("phases", {}).items()
            },
            solver=SolverConfig(**config_dict.get("solver", {})),
            time=TimeConfig(
                max_time=config_dict.get("time", {}).get("max_time", 1.0),
                save_interval=config_dict.get("time", {}).get("save_interval", 0.1),
            ),
            initial_condition=InitialConditionConfig(
                background_layer=config_dict.get("initial_conditions", {})
                .get("background", {})
                .get("height_fraction"),
                objects=[
                    ObjectConfig(**obj)
                    for obj in config_dict.get("initial_conditions", {}).get(
                        "objects", []
                    )
                ],
                velocity=config_dict.get("initial_conditions", {}).get(
                    "velocity", {"type": "zero"}
                ),
            ),
            output=OutputConfig(**config_dict.get("output", {})),
        )

    def save(self, filepath: str):
        """設定をYAMLファイルに保存

        Args:
            filepath: 保存先のパス
        """
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

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
