"""シミュレーションの初期状態を生成するモジュール"""

import numpy as np
from typing import Dict, Any
import logging

from physics.levelset import LevelSetParameters
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

    def _setup_levelset(
        self, dimensions: list, domain_size: list, initial_conditions: Dict
    ) -> np.ndarray:
        """初期界面を設定

        Args:
            dimensions: グリッドの次元
            domain_size: 領域のサイズ
            initial_conditions: 初期条件の設定

        Returns:
            レベルセット関数の値
        """
        # 背景相（水層）の設定
        background = initial_conditions.get("background", {})
        height_fraction = background.get("height_fraction", 0.8)
        water_height = height_fraction * domain_size[2]

        # z座標の生成
        z = np.linspace(0, domain_size[2], dimensions[2])
        Z = z.reshape(1, 1, -1)  # ブロードキャスト用に形状を変更

        # レベルセット関数の初期化
        # Z < water_heightの領域が水（正）、それ以外が窒素（負）
        levelset_data = water_height - Z  # 水層の上面からの符号付き距離
        levelset_data = np.broadcast_to(levelset_data, tuple(dimensions)).copy()

        # オブジェクト（窒素球）の追加
        objects = initial_conditions.get("objects", [])
        for obj in objects:
            if obj["type"] == "sphere":
                center = obj.get("center", [0.5, 0.5, 0.4])
                radius = obj.get("radius", 0.2)

                # メッシュグリッドを生成
                x = np.linspace(0, domain_size[0], dimensions[0])
                y = np.linspace(0, domain_size[1], dimensions[1])
                z = np.linspace(0, domain_size[2], dimensions[2])
                X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

                # 球の距離関数を計算（球の内部が負、外部が正）
                sphere_dist = (
                    np.sqrt(
                        (X - center[0] * domain_size[0]) ** 2
                        + (Y - center[1] * domain_size[1]) ** 2
                        + (Z - center[2] * domain_size[2]) ** 2
                    )
                    - radius * domain_size[0]
                )

                # レベルセット関数を更新
                # minを取ることで、窒素相（負の値）が水相（正の値）を上書き
                levelset_data = np.minimum(levelset_data, -sphere_dist)

        return levelset_data

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

        # シミュレーション状態を作成
        state = SimulationState(shape=tuple(dimensions), dx=dx)

        # Level Set関数の初期化
        level_set_params = LevelSetParameters(
            **self.config.get("numerical", {}).get("level_set", {})
        )
        state.levelset.params = level_set_params
        state.levelset.data = self._setup_levelset(
            dimensions, domain_size, self.config.get("initial_conditions", {})
        )

        # 速度場の初期化
        velocity_config = self.config.get("initial_conditions", {}).get("velocity", {})
        velocity_type = velocity_config.get("type", "zero")

        if velocity_type == "zero":
            for comp in state.velocity.components:
                comp.data.fill(0.0)
        # TODO: 他の初期速度場のタイプを追加

        # 圧力場の初期化
        state.pressure.data.fill(0.0)

        # 物性値マネージャーの設定
        state.properties = PropertiesManager(
            phase1=self.fluid_properties.get("water", FluidProperties(1000.0, 1.0e-3)),
            phase2=self.fluid_properties.get(
                "nitrogen", FluidProperties(1.25, 1.81e-5)
            ),
        )

        if self.logger:
            self.logger.info("初期状態の生成完了")
            self.logger.info(f"State summary: {state}")

        return state
