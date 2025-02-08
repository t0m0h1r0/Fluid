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
        """初期フィールドを設定"""
        # 物理パラメータ取得
        rho_w = self.fluid_properties["water"].density
        rho_a = self.fluid_properties["nitrogen"].density
        sigma = self.fluid_properties["water"].surface_tension
        g = self.config.get("physics", {}).get("gravity", 9.81)
        
        # グリッド生成
        x = np.linspace(0, domain_size[0], dimensions[0])
        y = np.linspace(0, domain_size[1], dimensions[1])
        z = np.linspace(0, domain_size[2], dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # 初期化
        levelset = np.zeros_like(X)
        pressure = np.zeros_like(X)
        velocity = [np.zeros_like(X) for _ in range(3)]
        
        # 水層の設定
        water_height = initial_conditions.get("background", {}).get("height_fraction", 0.8) * domain_size[2]
        levelset = Z - water_height  # 界面からの距離
        water_region = (levelset < 0)
        pressure[water_region] = rho_w * g * (water_height - Z[water_region])
        
        # 物体の追加（気泡や水球）
        for obj in initial_conditions.get("objects", []):
            if obj["type"] == "sphere":
                center = [c * d for c, d in zip(obj["center"], domain_size)]
                radius = obj["radius"] * domain_size[0]
                
                # 距離関数の計算
                sphere_dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2) - radius
                
                # 界面の更新
                levelset = np.where(sphere_dist < levelset, sphere_dist, levelset)
                
                # 圧力設定
                if center[2] < water_height:  # 水中の気泡
                    p_interface = rho_w * g * (water_height - center[2])
                    pressure[sphere_dist < 0] = p_interface + 2 * sigma / radius
                else:  # 空気中の水球
                    p_interface = rho_a * g * (domain_size[2] - center[2])
                    pressure[sphere_dist < 0] = p_interface - 2 * sigma / radius
        
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