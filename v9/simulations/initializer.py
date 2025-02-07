"""シミュレーションの初期化を管理するモジュール

このモジュールは、シミュレーションの初期状態を設定する機能を提供します。
"""

from typing import Dict, Any
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField, LevelSetParameters
from physics.properties import PropertiesManager, FluidProperties
from .state import SimulationState
from logging import SimulationLogger


class SimulationInitializer:
    """シミュレーションの初期化を管理するクラス

    計算領域の設定、初期場の生成、物性値の設定などを行います。
    """

    def __init__(self, config: Dict[str, Any], logger: SimulationLogger):
        """初期化マネージャーを初期化

        Args:
            config: シミュレーション設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger.start_section("init")

        # グリッド情報の計算
        self.dimensions = config["domain"]["dimensions"]
        self.domain_size = config["domain"]["size"]
        self.dx = self.domain_size[0] / self.dimensions[0]

        # Level Set用パラメータの設定
        self.ls_params = LevelSetParameters(
            epsilon=config["numerical"]["level_set"]["epsilon"],
            reinit_interval=config["numerical"]["level_set"]["reinit_interval"],
            reinit_steps=config["numerical"]["level_set"]["reinit_steps"],
        )

    def create_initial_state(self) -> SimulationState:
        """初期状態を生成

        Returns:
            初期化されたシミュレーション状態
        """
        self.logger.info("初期状態を生成中...")

        try:
            # フィールドの初期化
            velocity = self._create_velocity_field()
            pressure = self._create_pressure_field()
            levelset = self._create_levelset_field()
            properties = self._create_properties_manager()

            # 初期状態の生成
            state = SimulationState(
                velocity=velocity,
                pressure=pressure,
                levelset=levelset,
                properties=properties,
            )

            # 統計情報の初期化
            state.update_statistics()

            self.logger.info("初期状態の生成完了")
            return state

        except Exception as e:
            self.logger.log_error_with_context(
                "初期状態の生成中にエラーが発生",
                e,
                {"dimensions": self.dimensions, "dx": self.dx},
            )
            raise

    def _create_velocity_field(self) -> VectorField:
        """速度場を初期化"""
        velocity = VectorField(self.dimensions, self.dx)

        # 初期速度の設定（設定ファイルに基づく）
        if "initial_velocity" in self.config.get("initial_conditions", {}):
            vel_config = self.config["initial_conditions"]["initial_velocity"]
            if vel_config["type"] == "uniform":
                for i, v in enumerate(vel_config["value"]):
                    velocity.components[i].data.fill(v)
            elif vel_config["type"] == "function":
                # カスタム関数による初期化
                self._initialize_velocity_from_function(velocity, vel_config)

        return velocity

    def _create_pressure_field(self) -> ScalarField:
        """圧力場を初期化"""
        return ScalarField(self.dimensions, self.dx)

    def _create_levelset_field(self) -> LevelSetField:
        """Level Set場を初期化"""
        levelset = LevelSetField(self.dimensions, self.dx, self.ls_params)

        # 初期界面の設定
        for obj in self.config["initial_conditions"]["objects"]:
            if obj["type"] == "sphere":
                self._add_sphere(levelset, obj)
            elif obj["type"] == "layer":
                self._add_layer(levelset, obj)
            elif obj["type"] == "cylinder":
                self._add_cylinder(levelset, obj)

        # 再初期化
        levelset.reinitialize()
        return levelset

    def _create_properties_manager(self) -> PropertiesManager:
        """物性値マネージャーを初期化"""
        phase1 = FluidProperties(**self.config["physics"]["phases"]["water"])
        phase2 = FluidProperties(**self.config["physics"]["phases"]["air"])
        return PropertiesManager(phase1=phase1, phase2=phase2)

    def _add_sphere(self, levelset: LevelSetField, config: Dict[str, Any]):
        """球状の界面を追加"""
        center = np.array(config["center"])
        radius = config["radius"]

        x = np.linspace(0, self.domain_size[0], self.dimensions[0])
        y = np.linspace(0, self.domain_size[1], self.dimensions[1])
        z = np.linspace(0, self.domain_size[2], self.dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        distance = np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )

        if config["phase"] == "water":
            levelset.data = np.minimum(levelset.data, distance - radius)
        else:
            levelset.data = np.maximum(levelset.data, -(distance - radius))

    def _add_layer(self, levelset: LevelSetField, config: Dict[str, Any]):
        """層状の界面を追加"""
        normal = np.array(config["normal"])
        position = config["position"]

        x = np.linspace(0, self.domain_size[0], self.dimensions[0])
        y = np.linspace(0, self.domain_size[1], self.dimensions[1])
        z = np.linspace(0, self.domain_size[2], self.dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        distance = (
            normal[0] * X
            + normal[1] * Y
            + normal[2] * Z
            - position * self.domain_size[normal.argmax()]
        )

        if config["phase"] == "water":
            levelset.data = np.minimum(levelset.data, distance)
        else:
            levelset.data = np.maximum(levelset.data, -distance)

    def _add_cylinder(self, levelset: LevelSetField, config: Dict[str, Any]):
        """円柱状の界面を追加"""
        center = np.array(config["center"])
        radius = config["radius"]
        height = config["height"]
        axis = config.get("axis", 2)  # デフォルトはz軸方向

        x = np.linspace(0, self.domain_size[0], self.dimensions[0])
        y = np.linspace(0, self.domain_size[1], self.dimensions[1])
        z = np.linspace(0, self.domain_size[2], self.dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        coords = [X, Y, Z]

        # 軸方向以外の距離を計算
        radial_coords = coords.copy()
        radial_coords.pop(axis)
        center_radial = np.delete(center, axis)

        distance_radial = np.sqrt(
            sum((coord - c) ** 2 for coord, c in zip(radial_coords, center_radial))
        )

        # 軸方向の距離を計算
        axis_coord = coords[axis]
        axis_center = center[axis]
        distance_axis = np.abs(axis_coord - axis_center) - height / 2

        # 円柱からの距離場を構築
        distance = np.maximum(distance_radial - radius, distance_axis)

        if config["phase"] == "water":
            levelset.data = np.minimum(levelset.data, distance)
        else:
            levelset.data = np.maximum(levelset.data, -distance)

    def _initialize_velocity_from_function(
        self, velocity: VectorField, config: Dict[str, Any]
    ):
        """カスタム関数による速度場の初期化"""
        function_type = config["function"]
        if function_type == "vortex":
            # 渦状の速度場
            center = np.array(config.get("center", [x / 2 for x in self.domain_size]))
            strength = config.get("strength", 1.0)

            x = np.linspace(0, self.domain_size[0], self.dimensions[0])
            y = np.linspace(0, self.domain_size[1], self.dimensions[1])
            z = np.linspace(0, self.domain_size[2], self.dimensions[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            R = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            theta = np.arctan2(Y - center[1], X - center[0])

            velocity.components[0].data = -strength * np.sin(theta) * R
            velocity.components[1].data = strength * np.cos(theta) * R
            velocity.components[2].data = 0.0
