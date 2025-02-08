"""シミュレーションの初期状態を生成するモジュール"""

import numpy as np
from typing import Dict, Any, Tuple, List
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

    def _setup_initial_fields(
        self, dimensions: list, domain_size: list, initial_conditions: Dict
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """初期フィールドを設定

        Args:
            dimensions: グリッドの次元
            domain_size: 領域のサイズ
            initial_conditions: 初期条件の設定

        Returns:
            levelset, pressure, velocityのタプル
        """
        # 物理パラメータの取得
        water_props = self.fluid_properties.get("water")
        rho_w = water_props.density
        sigma = water_props.surface_tension
        g = self.config.get("physics", {}).get("gravity", 9.81)
        p_atm = self.config.get("physics", {}).get("atmospheric_pressure", 101325.0)

        # グリッドの生成
        x = np.linspace(0, domain_size[0], dimensions[0])
        y = np.linspace(0, domain_size[1], dimensions[1])
        z = np.linspace(0, domain_size[2], dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # 初期フィールドの作成
        levelset = np.zeros_like(X)
        pressure = np.full_like(X, p_atm)  # 大気圧で初期化
        velocity = [np.zeros_like(X) for _ in range(3)]  # 速度場はゼロで初期化

        # 背景相（水層）の設定
        background = initial_conditions.get("background", {})
        height_fraction = background.get("height_fraction", 0.8)
        water_height = height_fraction * domain_size[2]

        # 水層のLevel Setと圧力を設定
        levelset = water_height - Z  # 水面からの符号付き距離
        water_region = (levelset > 0)
        pressure[water_region] = p_atm + rho_w * g * (water_height - Z[water_region])

        # オブジェクト（窒素球）の追加
        objects = initial_conditions.get("objects", [])
        for obj in objects:
            if obj["type"] == "sphere":
                center = obj.get("center", [0.5, 0.5, 0.4])
                radius = obj.get("radius", 0.2)

                # 中心座標と半径を物理単位に変換
                center_phys = [c * d for c, d in zip(center, domain_size)]
                radius_phys = radius * domain_size[0]

                # 球の距離関数を計算
                sphere_dist = np.sqrt(
                    (X - center_phys[0]) ** 2 +
                    (Y - center_phys[1]) ** 2 +
                    (Z - center_phys[2]) ** 2
                ) - radius_phys

                # 球の内部を特定
                bubble_region = (sphere_dist < 0)
                if np.any(bubble_region):
                    # Level Set関数を更新
                    levelset[bubble_region] = sphere_dist[bubble_region]

                    # 気泡内の圧力を設定
                    # 界面での静水圧
                    p_interface = p_atm + rho_w * g * (water_height - center_phys[2])
                    # 表面張力による圧力ジャンプ（球なので曲率は2/R）
                    dp_laplace = 2 * sigma / radius_phys
                    # 気泡内圧力
                    p_bubble = p_interface + dp_laplace
                    pressure[bubble_region] = p_bubble

        return levelset, pressure, velocity

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

        # 初期フィールドの設定
        levelset_data, pressure_data, velocity_data = self._setup_initial_fields(
            dimensions, domain_size, self.config.get("initial_conditions", {})
        )

        # シミュレーション状態を作成
        state = SimulationState(shape=tuple(dimensions), dx=dx)

        # 各フィールドの設定
        state.levelset.params = level_set_params
        state.levelset.data = levelset_data
        state.pressure.data = pressure_data
        for i, comp in enumerate(state.velocity.components):
            comp.data = velocity_data[i]

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