"""シミュレーションの初期状態を生成するモジュール"""

import numpy as np
from typing import Dict, Any
import logging

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField, LevelSetParameters
from physics.properties import PropertiesManager, FluidProperties
from simulations.state import SimulationState


class SimulationInitializer:
    """シミュレーション初期状態の生成クラス"""

    def __init__(self, config: Dict[str, Any], logger=None):
        """初期化

        Args:
            config: シミュレーション設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # 相の物性値を設定
        self._setup_phases()

    def _setup_phases(self):
        """相の物性値を設定"""
        phases_config = self.config.get("physics", {}).get("phases", {})

        # デフォルトの物性値を設定
        if not phases_config:
            phases_config = {
                "water": {
                    "density": 1000.0,
                    "viscosity": 1.0e-3,
                    "surface_tension": 0.07,
                },
                "nitrogen": {
                    "density": 1.25,
                    "viscosity": 1.81e-5,
                    "surface_tension": 0.0,
                },
            }

        # FluidPropertiesインスタンスを作成
        self.fluid_properties = {}
        for phase_name, props in phases_config.items():
            self.fluid_properties[phase_name] = FluidProperties(
                density=props["density"],
                viscosity=props["viscosity"],
                surface_tension=props.get("surface_tension", 0.0),
            )

    def create_initial_state(self) -> SimulationState:
        """初期状態を生成

        Returns:
            初期化されたシミュレーション状態
        """
        if self.logger:
            self.logger.info("初期状態を生成中...")

        # 計算領域の設定
        domain_config = self.config.get("domain", {})
        dimensions = domain_config.get("dimensions", [64, 64, 64])
        domain_size = domain_config.get("size", [1.0, 1.0, 1.0])

        # グリッド間隔の計算
        dx = domain_size[0] / dimensions[0]

        # Level Set パラメータの設定
        level_set_params = LevelSetParameters(
            **self.config.get("numerical", {}).get("level_set", {})
        )

        # レベルセット場の初期化
        levelset = LevelSetField(
            shape=tuple(dimensions), dx=dx, params=level_set_params
        )

        # 初期界面の形状を設定
        initial_conditions = self.config.get("initial_conditions", {})

        # 背景相の高さ設定
        background_config = initial_conditions.get("background", {})
        background_phase = background_config.get("phase", "water")
        height_fraction = background_config.get("height_fraction", 0.75)

        # 初期のレベルセット関数を設定
        # 背景相の下部を正の値、上部を負の値に
        z = np.linspace(0, domain_size[2], dimensions[2])
        background_height = height_fraction * domain_size[2]

        levelset_data = np.zeros(tuple(dimensions))
        for k in range(dimensions[2]):
            levelset_data[:, :, k] = background_height - z[k]

        # オブジェクト（窒素球）の追加
        objects = initial_conditions.get("objects", [])
        for obj in objects:
            if obj["type"] == "sphere":
                center = obj.get("center", [0.5, 0.5, 0.5])
                radius = obj.get("radius", 0.2)

                # メッシュグリッドを生成
                x, y, z = np.meshgrid(
                    np.linspace(0, domain_size[0], dimensions[0]),
                    np.linspace(0, domain_size[1], dimensions[1]),
                    np.linspace(0, domain_size[2], dimensions[2]),
                )

                # 球の距離関数を計算
                sphere_dist = (
                    np.sqrt(
                        (
                            (x - center[0] * domain_size[0]) ** 2
                            + (y - center[1] * domain_size[1]) ** 2
                            + (z - center[2] * domain_size[2]) ** 2
                        )
                    )
                    - radius * domain_size[0]
                )

                # レベルセット関数を更新
                levelset_data = np.minimum(levelset_data, sphere_dist)

        # レベルセット場にデータを設定
        levelset.data = levelset_data

        # 速度場の初期化
        velocity_config = initial_conditions.get("velocity", {})
        velocity_type = velocity_config.get("type", "zero")

        velocity = VectorField(tuple(dimensions), dx)
        if velocity_type == "zero":
            for comp in velocity.components:
                comp.data.fill(0.0)

        # 圧力場の初期化
        pressure = ScalarField(tuple(dimensions), dx)
        pressure.data.fill(0.0)

        # 物性値マネージャーの初期化
        properties_manager = PropertiesManager(
            phase1=self.fluid_properties.get("water", FluidProperties(1000.0, 1.0e-3)),
            phase2=self.fluid_properties.get(
                "nitrogen", FluidProperties(1.25, 1.81e-5)
            ),
        )

        # シミュレーション状態を作成
        state = SimulationState(
            velocity=velocity,
            pressure=pressure,
            levelset=levelset,
            properties=properties_manager,
        )

        if self.logger:
            self.logger.info("初期状態の生成完了")

        return state
