"""シミュレーション設定を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml

from physics.levelset.initializer import Phase  # インポートを追加


class BoundaryType(Enum):
    """境界条件の種類を表す列挙型"""

    PERIODIC = "periodic"
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"


@dataclass
class PhaseConfig:
    """流体の物性値を保持するクラス"""

    density: float  # 密度 [kg/m³]
    viscosity: float  # 動粘性係数 [Pa·s]
    surface_tension: float = 0.0  # 表面張力係数 [N/m]

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.density <= 0:
            raise ValueError("密度は正の値である必要があります")
        if self.viscosity <= 0:
            raise ValueError("粘性は正の値である必要があります")
        if self.surface_tension < 0:
            raise ValueError("表面張力は非負である必要があります")


@dataclass
class DomainConfig:
    """計算領域の設定を保持するクラス"""

    dimensions: List[int]  # [nx, ny, nz]の格子点数
    size: List[float]  # [Lx, Ly, Lz]の物理サイズ [m]

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if not all(isinstance(dim, int) and dim > 0 for dim in self.dimensions):
            raise ValueError("格子点数は正の整数である必要があります")
        if not all(isinstance(size, (int, float)) and size > 0 for size in self.size):
            raise ValueError("領域サイズは正の値である必要があります")
        if len(self.dimensions) != len(self.size):
            raise ValueError("dimensionsとsizeは同じ次元数である必要があります")
        if not (2 <= len(self.dimensions) <= 3):
            raise ValueError("2次元または3次元である必要があります")


@dataclass
class BoundaryConfig:
    """境界条件の設定を保持するクラス"""

    x: Dict[str, str]  # left, right
    y: Dict[str, str]  # front, back
    z: Dict[str, str]  # bottom, top

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        valid_types = set(t.value for t in BoundaryType)
        for direction in [self.x, self.y, self.z]:
            for boundary_type in direction.values():
                if boundary_type not in valid_types:
                    raise ValueError(f"無効な境界条件: {boundary_type}")


@dataclass
class InterfaceConfig:
    """界面の設定を保持するクラス"""

    phase: Phase
    object_type: str  # "background", "layer", "sphere"
    height: Optional[float] = None  # レイヤー用
    center: Optional[List[float]] = None  # 球体用
    radius: Optional[float] = None  # 球体用

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.object_type == "background":
            if any([self.height, self.center, self.radius]):
                raise ValueError("背景相には高さ、中心、半径は指定できません")
        elif self.object_type == "layer":
            if self.height is not None and not 0 <= self.height:
                raise ValueError("高さは正である必要があります")
            if any([self.center, self.radius]):
                raise ValueError("レイヤーには高さのみ指定してください")
        elif self.object_type == "sphere":
            if not self.center or len(self.center) != 3:
                raise ValueError("球体には3次元の中心座標が必要です")
            if not self.radius or self.radius <= 0:
                raise ValueError("球体には正の半径が必要です")
            if self.height is not None:
                raise ValueError("球体には高さは指定できません")
        else:
            raise ValueError(f"未対応のオブジェクトタイプ: {self.object_type}")


@dataclass
class InitialConditionConfig:
    """初期条件の設定を保持するクラス"""

    velocity: Dict[str, str]  # 初期速度場の設定
    background: Dict[str, Any]  # 背景相の設定
    objects: List[Dict[str, Any]] = field(
        default_factory=list
    )  # 界面オブジェクトのリスト

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if "phase" not in self.background:
            raise ValueError("背景相には相の指定が必要です")
        if "type" not in self.velocity:
            raise ValueError("初期速度場の種類の指定が必要です")

    def get(self, key: str, default=None):
        """辞書のようなアクセスを可能にするメソッド"""
        if hasattr(self, key):
            return getattr(self, key)
        return default


@dataclass
class OutputConfig:
    """出力の設定を保持するクラス"""

    output_dir: str = "results"
    format: str = "png"
    dpi: int = 300
    colormap: str = "viridis"
    show_colorbar: bool = True
    show_axes: bool = True
    show_grid: bool = False
    slices: Dict[str, List[Any]] = field(
        default_factory=lambda: {"axes": ["xy", "xz", "yz"], "positions": [0.5]}
    )
    fields: Dict[str, Dict[str, bool]] = field(
        default_factory=lambda: {
            "velocity": {"enabled": True},
            "pressure": {"enabled": True},
            "levelset": {"enabled": True},
        }
    )

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if not self.output_dir:
            raise ValueError("output_dirは空にできません")
        if self.dpi <= 0:
            raise ValueError("dpiは正の値である必要があります")
        valid_axes = {"xy", "xz", "yz"}
        if not all(axis in valid_axes for axis in self.slices["axes"]):
            raise ValueError(f"無効なスライス軸。有効な値: {valid_axes}")
        if not all(0 <= pos <= 1 for pos in self.slices["positions"]):
            raise ValueError("スライス位置は0から1の間である必要があります")


@dataclass
class SimulationConfig:
    """シミュレーション全体の設定を保持するクラス"""

    domain: DomainConfig
    phases: Dict[str, PhaseConfig]
    boundary_conditions: BoundaryConfig
    initial_conditions: InitialConditionConfig
    numerical: Optional[Dict[str, Any]] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)
    interfaces: List[InterfaceConfig] = field(default_factory=list)

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        self.domain.validate()
        for phase in self.phases.values():
            phase.validate()
        self.boundary_conditions.validate()
        self.initial_conditions.validate()
        self.output.validate()

        # インターフェースの検証を追加
        for interface in self.interfaces:
            interface.validate()

    @classmethod
    def from_yaml(cls, filepath: str) -> "SimulationConfig":
        """YAMLファイルから設定を読み込む"""
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)

        # Phase enumとYAMLの文字列のマッピング
        phase_mapping = {"water": Phase.PHASE_1, "nitrogen": Phase.PHASE_2}

        # 必要な設定を読み込む
        domain = DomainConfig(
            dimensions=config_dict["domain"]["dimensions"],
            size=[float(s) for s in config_dict["domain"]["size"]],
        )

        phases = {
            name: PhaseConfig(**props) for name, props in config_dict["phases"].items()
        }

        boundary_conditions = BoundaryConfig(**config_dict["boundary_conditions"])

        initial_conditions = InitialConditionConfig(**config_dict["initial_conditions"])

        # background の情報を取得
        background_config = initial_conditions.background

        # background の InterfaceConfig を作成
        background_interface = InterfaceConfig(
            phase=phase_mapping[background_config["phase"]],
            object_type="background",
        )

        # interfaces リストの先頭に background を追加
        interfaces = [background_interface] + [
            InterfaceConfig(
                phase=phase_mapping[obj["phase"]],
                object_type=obj["type"],
                height=obj.get("height", None),
                center=obj.get("center", None),
                radius=obj.get("radius", None),
            )
            for obj in initial_conditions.get("objects", [])
        ]

        # 追加の設定
        numerical = config_dict.get("numerical", {})
        output = OutputConfig(**config_dict.get("output", {}))

        # SimulationConfigの生成
        config = cls(
            domain=domain,
            phases=phases,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            numerical=numerical,
            output=output,
            interfaces=interfaces,
        )

        # 設定の妥当性を検証
        config.validate()

        return config
