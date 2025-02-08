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
        self.fluid_properties = self._setup_fluid_properties()

    def _setup_fluid_properties(self) -> Dict[str, FluidProperties]:
        """流体の物性値を設定

        Returns:
            物性値のディクショナリ
        """
        phases_config = self.config.get("physics", {}).get("phases", {
            "water": {
                "density": 1000.0,
                "viscosity": 1.0e-3,
                "surface_tension": 0.07,
            },
            "nitrogen": {
                "density": 1.25,
                "viscosity": 1.81e-5,
                "surface_tension": 0.0,
            }
        })

        return {
            phase_name: FluidProperties(
                density=props.get("density", 1000.0),
                viscosity=props.get("viscosity", 1.0e-3),
                surface_tension=props.get("surface_tension", 0.0)
            )
            for phase_name, props in phases_config.items()
        }

    def _compute_signed_distance(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        Z: np.ndarray, 
        center: List[float], 
        radius: float
    ) -> np.ndarray:
        """符号付き距離関数を計算

        Args:
            X, Y, Z: 物理空間の座標グリッド
            center: 球の中心座標
            radius: 球の半径

        Returns:
            符号付き距離関数（水：正、窒素：負）
        """
        # 中心からの距離を計算
        distance = np.sqrt(
            (X - center[0]) ** 2 +
            (Y - center[1]) ** 2 +
            (Z - center[2]) ** 2
        )
        
        # 距離関数を計算（球の内部は負、外部は正）
        return radius - distance

    def _setup_initial_fields(
        self, 
        dimensions: List[int], 
        domain_size: List[float], 
        initial_conditions: Dict
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
        nitrogen_props = self.fluid_properties.get("nitrogen")
        
        # プロパティが見つからない場合のデフォルト値
        rho_w = water_props.density if water_props else 1000.0
        rho_n = nitrogen_props.density if nitrogen_props else 1.25
        sigma = water_props.surface_tension if water_props else 0.07
        g = self.config.get("physics", {}).get("gravity", 9.81)

        # 座標グリッドの生成
        x = np.linspace(0, domain_size[0], dimensions[0])
        y = np.linspace(0, domain_size[1], dimensions[1])
        z = np.linspace(0, domain_size[2], dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # 初期フィールドの作成
        levelset = np.zeros_like(X)
        pressure = np.zeros_like(X)
        velocity = [np.zeros_like(X) for _ in range(3)]

        # 背景相の設定
        background = initial_conditions.get("background", {})
        height_fraction = background.get("height_fraction", 0.8)
        water_height = height_fraction * domain_size[2]

        # 背景相のLevel Set関数を設定（水：正、窒素：負）
        levelset = Z - water_height

        # オブジェクト（窒素球）の追加
        objects = initial_conditions.get("objects", [])
        for obj in objects:
            if obj["type"] == "sphere":
                center = obj.get("center", [0.5, 0.5, 0.4])
                radius = obj.get("radius", 0.2)

                # 中心座標と半径を物理的な寸法に変換
                center_phys = [c * d for c, d in zip(center, domain_size)]
                radius_phys = radius * domain_size[0]

                # Level Set関数の計算（球の内部は負の値）
                sphere_dist = self._compute_signed_distance(X, Y, Z, center_phys, radius_phys)

                # 球の内部のLevelSetを更新
                mask = sphere_dist > 0
                levelset[mask] = -sphere_dist[mask]

                # 球の中心点を特定
                center_indices = np.unravel_index(
                    np.argmin(np.abs(Z - center_phys[2])), 
                    Z.shape
                )

                # 圧力場の設定
                # 1. 空気層の圧力: Pa(z) = ρn * g * (zmax - z)
                p_air = rho_n * g * (domain_size[2] - Z)

                # 2. 水層の圧力 
                # P界面 = ρn * g * (zmax - z界面)
                p_interface = rho_n * g * (domain_size[2] - water_height)
                
                # Pw(z) = P界面 + ρw * g * (z界面 - z)
                p_water = p_interface + rho_w * g * (water_height - Z)

                # 3. 窒素球内の圧力: Pn = Pw(z球) + 2σ/R
                p_sphere_location = p_water[center_indices]
                dp_laplace = 2 * sigma / radius_phys  # 曲率による圧力ジャンプ
                p_bubble = p_sphere_location + dp_laplace

                # 圧力場の設定
                pressure[Z < water_height] = p_water[Z < water_height]  # 水層
                pressure[Z >= water_height] = p_air[Z >= water_height]  # 空気層
                pressure[mask] = p_bubble  # 球の内部

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
            phase2=self.fluid_properties.get("nitrogen", FluidProperties(1.25, 1.81e-5)),
        )

        if self.logger:
            self.logger.info("初期状態の生成完了")
            self.logger.info(f"State summary: {state}")

        return state