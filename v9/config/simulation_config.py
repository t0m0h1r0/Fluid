"""シミュレーション設定を管理するモジュール

このモジュールは、YAMLフォーマットの設定ファイルを読み込み、
適切なクラスのインスタンスに変換する機能を提供します。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from physics.properties import FluidProperties


@dataclass
class DomainConfig:
    """計算領域の設定

    Attributes:
        nx, ny, nz: 各方向のグリッド数
        lx, ly, lz: 各方向の領域サイズ [m]
    """

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float

    def validate(self):
        """設定値の妥当性を検証"""
        if any(n <= 0 for n in [self.nx, self.ny, self.nz]):
            raise ValueError("グリッド数は正の整数である必要があります")
        if any(l <= 0 for l in [self.lx, self.ly, self.lz]):
            raise ValueError("領域サイズは正の値である必要があります")


@dataclass
class SolverConfig:
    """ソルバーの設定

    Attributes:
        time_integrator: 時間積分法の種類
        pressure_solver: 圧力ソルバーの種類と設定
        convergence_criteria: 収束判定基準
    """

    time_integrator: str = "rk4"
    pressure_solver: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "sor",
            "omega": 1.5,
            "max_iterations": 100,
            "tolerance": 1e-6,
        }
    )
    convergence_criteria: Dict[str, float] = field(
        default_factory=lambda: {"velocity": 1e-6, "pressure": 1e-6}
    )
    use_weno: bool = True

    def validate(self):
        """設定値の妥当性を検証"""
        valid_integrators = ["euler", "rk4"]
        if self.time_integrator not in valid_integrators:
            raise ValueError(f"未対応の時間積分法です: {self.time_integrator}")

        valid_solvers = ["jacobi", "gauss_seidel", "sor"]
        if self.pressure_solver["type"] not in valid_solvers:
            raise ValueError(
                f"未対応の圧力ソルバーです: {self.pressure_solver['type']}"
            )


@dataclass
class TimeConfig:
    """時間発展の設定

    Attributes:
        dt: 初期時間刻み幅 [s]
        max_time: 最大計算時間 [s]
        cfl: CFL数
    """

    dt: float
    max_time: float
    cfl: float = 0.5
    save_interval: float = 0.1

    def validate(self):
        """設定値の妥当性を検証"""
        if self.dt <= 0 or self.max_time <= 0:
            raise ValueError("時間は正の値である必要があります")
        if self.cfl <= 0 or self.cfl > 1:
            raise ValueError("CFL数は0から1の間である必要があります")


@dataclass
class BoundaryConfig:
    """境界条件の設定

    Attributes:
        type: 境界条件の種類
        value: 境界値（必要な場合）
    """

    type: str
    value: Optional[float] = None

    def validate(self):
        """設定値の妥当性を検証"""
        valid_types = ["periodic", "dirichlet", "neumann"]
        if self.type not in valid_types:
            raise ValueError(f"未対応の境界条件です: {self.type}")


@dataclass
class PhaseConfig:
    """相の設定

    Attributes:
        density: 密度 [kg/m³]
        viscosity: 粘性係数 [Pa·s]
        surface_tension: 表面張力係数 [N/m]
    """

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
class InitialConditionConfig:
    """初期条件の設定

    Attributes:
        type: 初期条件の種類
        parameters: 各種パラメータ
    """

    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def validate(self):
        """設定値の妥当性を検証"""
        valid_types = ["droplet", "bubble", "layer", "custom"]
        if self.type not in valid_types:
            raise ValueError(f"未対応の初期条件です: {self.type}")


@dataclass
class OutputConfig:
    """出力の設定

    Attributes:
        directory: 出力ディレクトリ
        format: 出力フォーマット
        variables: 出力する変数のリスト
    """

    directory: str = "output"
    format: str = "vti"
    save_interval: float = 0.1
    variables: List[str] = field(
        default_factory=lambda: ["velocity", "pressure", "levelset"]
    )

    def validate(self):
        """設定値の妥当性を検証"""
        valid_formats = ["vti", "vtu", "hdf5"]
        if self.format not in valid_formats:
            raise ValueError(f"未対応の出力フォーマットです: {self.format}")


class SimulationConfig:
    """シミュレーション全体の設定

    YAMLファイルから設定を読み込み、各コンポーネントの
    設定クラスのインスタンスに変換します。
    """

    def __init__(self, config_file: str):
        """
        Args:
            config_file: 設定ファイルのパス
        """
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_file}")

        # 設定の読み込みと解析
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 各コンポーネントの設定を解析
        self.domain = self._parse_domain(config.get("domain", {}))
        self.solver = self._parse_solver(config.get("solver", {}))
        self.time = self._parse_time(config.get("time", {}))
        self.phases = self._parse_phases(config.get("phases", {}))
        self.boundary = self._parse_boundary(config.get("boundary", {}))
        self.initial = self._parse_initial(config.get("initial_condition", {}))
        self.output = self._parse_output(config.get("output", {}))

        # 全体の妥当性検証
        self.validate()

    def _parse_domain(self, config: Dict) -> DomainConfig:
        """領域設定の解析"""
        domain = DomainConfig(
            nx=config.get("nx", 64),
            ny=config.get("ny", 64),
            nz=config.get("nz", 64),
            lx=config.get("lx", 1.0),
            ly=config.get("ly", 1.0),
            lz=config.get("lz", 1.0),
        )
        domain.validate()
        return domain

    def _parse_solver(self, config: Dict) -> SolverConfig:
        """ソルバー設定の解析"""
        solver = SolverConfig(
            time_integrator=config.get("time_integrator", "rk4"),
            pressure_solver=config.get("pressure_solver", {}),
            convergence_criteria=config.get("convergence_criteria", {}),
            use_weno=config.get("use_weno", True),
        )
        solver.validate()
        return solver

    def _parse_time(self, config: Dict) -> TimeConfig:
        """時間設定の解析"""
        time = TimeConfig(
            dt=config.get("dt", 0.001),
            max_time=config.get("max_time", 1.0),
            cfl=config.get("cfl", 0.5),
            save_interval=config.get("save_interval", 0.1),
        )
        time.validate()
        return time

    def _parse_phases(self, config: Dict) -> Dict[str, PhaseConfig]:
        """相設定の解析"""
        phases = {}
        for name, props in config.items():
            phases[name] = PhaseConfig(
                density=props["density"],
                viscosity=props["viscosity"],
                surface_tension=props.get("surface_tension"),
            )
        return phases

    def _parse_boundary(self, config: Dict) -> Dict[str, BoundaryConfig]:
        """境界条件の解析"""
        boundaries = {}
        for direction, bc in config.items():
            if isinstance(bc, str):
                boundaries[direction] = BoundaryConfig(type=bc)
            else:
                boundaries[direction] = BoundaryConfig(
                    type=bc["type"], value=bc.get("value")
                )
            boundaries[direction].validate()
        return boundaries

    def _parse_initial(self, config: Dict) -> InitialConditionConfig:
        """初期条件の解析"""
        initial = InitialConditionConfig(
            type=config["type"], parameters=config.get("parameters", {})
        )
        initial.validate()
        return initial

    def _parse_output(self, config: Dict) -> OutputConfig:
        """出力設定の解析"""
        output = OutputConfig(
            directory=config.get("directory", "output"),
            format=config.get("format", "vti"),
            save_interval=config.get("save_interval", 0.1),
            variables=config.get("variables", []),
        )
        output.validate()
        return output

    def validate(self):
        """設定全体の妥当性を検証"""
        # 相の数のチェック
        if len(self.phases) < 1:
            raise ValueError("少なくとも1つの相が必要です")

        # 境界条件の整合性チェック
        required_directions = ["x", "y", "z"]
        if not all(d in self.boundary for d in required_directions):
            raise ValueError("全ての方向の境界条件が必要です")

    def save(self, filename: str):
        """設定をYAMLファイルとして保存"""
        config = {
            "domain": self.domain.__dict__,
            "solver": self.solver.__dict__,
            "time": self.time.__dict__,
            "phases": {name: phase.__dict__ for name, phase in self.phases.items()},
            "boundary": {dir: bc.__dict__ for dir, bc in self.boundary.items()},
            "initial_condition": self.initial.__dict__,
            "output": self.output.__dict__,
        }

        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)
