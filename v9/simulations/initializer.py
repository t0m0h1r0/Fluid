"""シミュレーションの初期化を担当するモジュール

このモジュールは、二相流シミュレーションの初期状態を設定します。
設定ファイルに基づいて、速度場、圧力場、界面関数などを
適切に初期化します。
"""

import numpy as np

from core.field import VectorField, ScalarField
from physics.multiphase import InterfaceOperations
from .config import SimulationConfig
from .state import SimulationState


class SimulationInitializer:
    """シミュレーション初期化クラス"""

    def __init__(self, config: SimulationConfig):
        """初期化クラスを構築

        Args:
            config: シミュレーション設定
        """
        self.config = config
        self._validate_config()

        # グリッド間隔の計算
        self.dx = np.array(
            [
                size / (dim - 1)
                for size, dim in zip(
                    self.config.domain.size, self.config.domain.dimensions
                )
            ]
        )

        # InterfaceOperationsの初期化
        self._interface_ops = InterfaceOperations(
            dx=self.dx,
            epsilon=self.config.numerical.get("interface", {}).get("epsilon", 1e-2),
        )

        # 追加の初期化パラメータ
        self._init_parameters = {
            "time": 0.0,
            "shape": tuple(
                list(self.config.domain.dimensions)
                + [len(self.config.domain.dimensions)]
            ),
        }

    def _validate_config(self):
        """設定の妥当性を検証"""
        if not self.config.domain:
            raise ValueError("領域設定が存在しません")
        if not self.config.physics:
            raise ValueError("物理設定が存在しません")
        if not self.config.initial_conditions:
            raise ValueError("初期条件が存在しません")

    def create_initial_state(self) -> SimulationState:
        """初期状態を生成

        Returns:
            初期化されたシミュレーション状態
        """
        # 基本フィールドの初期化
        velocity = VectorField(self._init_parameters["shape"], self.dx)
        levelset = ScalarField(self._init_parameters["shape"][:-1], self.dx)
        pressure = ScalarField(self._init_parameters["shape"][:-1], self.dx)

        # 速度場の初期化
        self._initialize_velocity(velocity)

        # 界面関数の初期化
        self._initialize_interface(levelset)

        # 状態の構築
        state = SimulationState(
            time=self._init_parameters["time"],
            velocity=velocity,
            levelset=levelset,
            pressure=pressure,
        )

        return state

    def _initialize_velocity(self, velocity: VectorField) -> None:
        """速度場を初期化

        Args:
            velocity: 初期化対象の速度場
        """
        velocity_config = self.config.initial_conditions.velocity

        if velocity_config["type"] == "zero":
            # ゼロ速度場（デフォルト）
            for comp in velocity.components:
                comp.data.fill(0.0)
        elif velocity_config["type"] == "uniform":
            # 一様流れ
            direction = velocity_config.get("direction", [1.0, 0.0, 0.0])
            magnitude = velocity_config.get("magnitude", 1.0)
            for i, comp in enumerate(velocity.components):
                comp.data.fill(direction[i] * magnitude)
        elif velocity_config["type"] == "vortex":
            # 渦流れ
            center = velocity_config.get("center", [0.5, 0.5, 0.5])
            strength = velocity_config.get("strength", 1.0)
            coords = np.meshgrid(
                *[np.linspace(0, 1, s) for s in velocity.shape[:-1]], indexing="ij"
            )
            r = np.sqrt(
                sum((c - cent) ** 2 for c, cent in zip(coords[:-1], center[:-1]))
            )
            velocity.components[0].data = -strength * (coords[1] - center[1]) / r
            velocity.components[1].data = strength * (coords[0] - center[0]) / r

    def _initialize_interface(self, levelset: ScalarField) -> None:
        """界面関数を初期化

        Args:
            levelset: 初期化対象の界面関数
        """
        # 界面設定の取得
        initial_conditions = self.config.initial_conditions
        background_phase = initial_conditions.background.get("phase")  # 背景相の取得

        # 最初のオブジェクトに基づいて界面を生成
        objects = initial_conditions.objects if initial_conditions.objects else []
        if objects:
            first_obj = objects[0]
            if first_obj.get("type") == "plate":
                height = first_obj.get("height", 0.5)
                phase = first_obj.get(
                    "phase", background_phase
                )  # オブジェクトの相を取得
                if phase == background_phase:
                    normal = [0, 0, 1]  # 背景相と同じ場合、正の高さ
                else:
                    normal = [0, 0, -1]  # 背景相と異なる場合、負の高さ
                point = [0.5, 0.5, height]
                levelset_data = self._interface_ops.create_plane(
                    shape=levelset.shape, normal=normal, point=point
                )
                levelset.data = levelset_data.data
            elif first_obj.get("type") == "sphere":
                center = first_obj.get("center", [0.5, 0.5, 0.5])
                radius = first_obj.get("radius", 0.1)
                phase = first_obj.get(
                    "phase", background_phase
                )  # オブジェクトの相を取得
                sign = (
                    1.0 if phase != background_phase else -1.0
                )  # 背景相と異なる場合は正の距離関数
                levelset_data = sign * self._interface_ops.create_sphere(
                    shape=levelset.shape, center=center, radius=radius
                )
                levelset.data = levelset_data.data
        else:
            # フォールバック: デフォルトの平面界面
            levelset_data = self._interface_ops.create_plane(
                shape=levelset.shape, normal=[0, 0, 1], point=[0.5, 0.5, 0.5]
            )
            levelset.data = levelset_data.data

        # 残りのオブジェクトを組み合わせる
        for obj in objects[1:]:
            if obj.get("type") == "plate":
                height = obj.get("height", 0.5)
                phase = obj.get("phase", background_phase)  # オブジェクトの相を取得
                if phase == background_phase:
                    normal = [0, 0, 1]
                else:
                    normal = [0, 0, -1]
                point = [0.5, 0.5, height]
                plate = self._interface_ops.create_plane(
                    shape=levelset.shape, normal=normal, point=point
                )
                levelset_data = self._interface_ops.combine_interfaces(
                    levelset, plate, "union"
                )
                levelset.data = levelset_data.data
            elif obj.get("type") == "sphere":
                center = obj.get("center", [0.5, 0.5, 0.5])
                radius = obj.get("radius", 0.1)
                phase = obj.get("phase", background_phase)  # オブジェクトの相を取得
                sign = 1.0 if phase != background_phase else -1.0
                sphere = sign * self._interface_ops.create_sphere(
                    shape=levelset.shape, center=center, radius=radius
                )
                levelset_data = self._interface_ops.combine_interfaces(
                    levelset, sphere, "union"
                )
                levelset.data = levelset_data.data
