"""シミュレーション設定を管理するモジュール"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto
import yaml


class Phase(Enum):
    """流体の相を表す列挙型"""

    WATER = auto()
    NITROGEN = auto()


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
class LevelSetConfig:
    """レベルセット法の設定を保持するクラス"""

    epsilon: float = 1.0e-2  # 界面の厚さ
    reinit_interval: int = 5  # 再初期化の間隔
    reinit_steps: int = 2  # 再初期化のステップ数

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.epsilon <= 0:
            raise ValueError("epsilonは正の値である必要があります")
        if self.reinit_interval <= 0:
            raise ValueError("reinit_intervalは正の値である必要があります")
        if self.reinit_steps <= 0:
            raise ValueError("reinit_stepsは正の値である必要があります")


@dataclass
class NumericalConfig:
    """数値計算の設定を保持するクラス"""

    time_integrator: str = "euler"
    max_time: float = 2.0
    initial_dt: float = 0.001
    save_interval: float = 0.01
    cfl: float = 0.5
    level_set: LevelSetConfig = field(default_factory=LevelSetConfig)

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.time_integrator not in ["euler", "rk4"]:
            raise ValueError("time_integratorはeulerまたはrk4である必要があります")
        if self.max_time <= 0:
            raise ValueError("max_timeは正の値である必要があります")
        if self.initial_dt <= 0:
            raise ValueError("initial_dtは正の値である必要があります")
        if self.save_interval <= 0:
            raise ValueError("save_intervalは正の値である必要があります")
        if not 0 < self.cfl <= 1:
            raise ValueError("cflは0から1の間である必要があります")
        self.level_set.validate()


@dataclass
class InterfaceConfig:
    """界面の設定を保持するクラス"""

    phase: Phase
    object_type: str  # "background", "layer", "sphere"
    height_fraction: Optional[float] = None  # レイヤー用
    center: Optional[List[float]] = None  # 球体用
    radius: Optional[float] = None  # 球体用

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if self.object_type == "background":
            if any([self.height_fraction, self.center, self.radius]):
                raise ValueError("背景相には高さ、中心、半径は指定できません")
        elif self.object_type == "layer":
            if not self.height_fraction or not 0 <= self.height_fraction <= 1:
                raise ValueError("レイヤーには0から1の間の高さが必要です")
            if any([self.center, self.radius]):
                raise ValueError("レイヤーには高さのみ指定してください")
        elif self.object_type == "sphere":
            if not self.center or len(self.center) != 3:
                raise ValueError("球体には3次元の中心座標が必要です")
            if not self.radius or self.radius <= 0:
                raise ValueError("球体には正の半径が必要です")
            if self.height_fraction is not None:
                raise ValueError("球体には高さは指定できません")
        else:
            raise ValueError(f"未対応のオブジェクトタイプ: {self.object_type}")


@dataclass
class InitialConditionConfig:
    """初期条件の設定を保持するクラス"""

    background: Dict[str, Any]  # 背景相の設定
    objects: List[Dict[str, Any]]  # 界面オブジェクトのリスト
    velocity: Dict[str, str]  # 初期速度場の設定

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        if "phase" not in self.background:
            raise ValueError("背景相には相の指定が必要です")
        if "type" not in self.velocity:
            raise ValueError("初期速度場の種類の指定が必要です")


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
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """設定値の妥当性を検証"""
        self.domain.validate()
        for phase in self.phases.values():
            phase.validate()
        self.boundary_conditions.validate()
        self.initial_conditions.validate()
        self.numerical.validate()
        self.output.validate()

    @classmethod
    def from_yaml(cls, filepath: str) -> "SimulationConfig":
        """YAMLファイルから設定を読み込む

        Args:
            filepath: 設定ファイルのパス

        Returns:
            SimulationConfig: 読み込まれた設定

        Raises:
            ValueError: 設定値が不正な場合
        """
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)

        # ドメイン設定の読み込み
        # サイズをリストに変換（辞書の場合はvaluesを使用）
        domain_size = config_dict["domain"]["size"]
        if isinstance(domain_size, dict):
            size_list = [float(domain_size[k]) for k in ["x", "y", "z"]]
        else:
            size_list = [float(s) for s in domain_size]

        domain = DomainConfig(
            dimensions=config_dict["domain"]["dimensions"], size=size_list
        )

        # 相の物性値を設定
        phases = {
            name: PhaseConfig(**props) for name, props in config_dict["phases"].items()
        }

        # 境界条件の設定
        boundary_conditions = BoundaryConfig(**config_dict["boundary_conditions"])

        # 初期条件の設定
        initial_conditions = InitialConditionConfig(**config_dict["initial_conditions"])

        # 数値計算の設定
        numerical = NumericalConfig(**config_dict.get("numerical", {}))
        if "level_set" in config_dict.get("numerical", {}):
            numerical.level_set = LevelSetConfig(
                **config_dict["numerical"]["level_set"]
            )

        # 出力設定
        output = OutputConfig(**config_dict.get("output", {}))

        # SimulationConfigの生成
        config = cls(
            domain=domain,
            phases=phases,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            numerical=numerical,
            output=output,
        )

        # 設定の妥当性を検証
        config.validate()

        return config
