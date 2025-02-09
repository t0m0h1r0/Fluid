"""シミュレーションの初期状態を生成するモジュール"""

import numpy as np
from typing import Dict, Any, Tuple, List
import logging

from physics.levelset import LevelSetParameters
from physics.properties import PropertiesManager, FluidProperties
from simulations.state import SimulationState


class SphereConfig:
    """球体の設定を保持するクラス"""
    def __init__(self, center: List[float], radius: float, is_water: bool):
        self.center = center
        self.radius = radius
        self.is_water = is_water


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
        self.fluid_properties = self._setup_fluid_properties()

    def _setup_fluid_properties(self) -> Dict[str, FluidProperties]:
        """流体の物性値を設定

        Returns:
            物性値のディクショナリ
        """
        phases_config = self.config.get("physics", {}).get(
            "phases",
            {
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
            },
        )

        return {
            phase_name: FluidProperties(
                density=props.get("density", 1000.0),
                viscosity=props.get("viscosity", 1.0e-3),
                surface_tension=props.get("surface_tension", 0.0),
            )
            for phase_name, props in phases_config.items()
        }

    def _compute_signed_distance_to_plane(
        self,
        Z: np.ndarray,
        height: float,
    ) -> np.ndarray:
        """平面からの符号付き距離を計算

        Args:
            Z: Z座標の配列（高さ方向）
            height: 平面の高さ

        Returns:
            符号付き距離（水領域が正）
        """
        return height - Z

    def _compute_signed_distance_to_sphere(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        center: List[float],
        radius: float,
    ) -> np.ndarray:
        """球からの符号付き距離を計算

        Args:
            X, Y, Z: 座標グリッド
            center: 球の中心座標
            radius: 球の半径

        Returns:
            符号付き距離（球の内部が正）
        """
        return radius - np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )

    def _compute_closest_interface_distance(
        self,
        Z: np.ndarray,
        water_height: float,
        spheres: List[SphereConfig],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """最も近い界面からの符号付き距離とフェーズインジケータを計算

        Args:
            Z: Z座標の配列（高さ方向）
            water_height: 水面の高さ
            spheres: 球体のリスト
            X, Y: X, Y座標の配列

        Returns:
            (符号付き距離, フェーズインジケータ)のタプル
            フェーズインジケータ: True=水, False=空気
        """
        # 初期化（大きな値で）
        min_distance = np.full_like(Z, np.inf)
        is_water = np.full_like(Z, True, dtype=bool)

        # 水面からの距離を計算
        plane_distance = self._compute_signed_distance_to_plane(Z, water_height)
        water_indicator = plane_distance > 0

        min_distance = np.abs(plane_distance)
        is_water = water_indicator

        # 各球体からの距離を計算
        for sphere in spheres:
            sphere_distance = self._compute_signed_distance_to_sphere(
                X, Y, Z, sphere.center, sphere.radius
            )
            
            # 距離の絶対値を比較
            abs_sphere_distance = np.abs(sphere_distance)
            closer_to_sphere = abs_sphere_distance < np.abs(min_distance)
            
            # より近い界面の情報で更新
            min_distance = np.where(closer_to_sphere, abs_sphere_distance, min_distance)
            
            # 球の内部か外部かで水/空気を判定
            sphere_phase = np.where(sphere_distance > 0,
                                  sphere.is_water,
                                  not sphere.is_water)
            
            is_water = np.where(closer_to_sphere, sphere_phase, is_water)

        # 符号を設定（水領域が正、空気領域が負）
        signed_distance = np.where(is_water, min_distance, -min_distance)

        return signed_distance, is_water

    def _compute_curvature_and_pressure_jump(
        self,
        levelset: np.ndarray,
        spheres: List[SphereConfig],
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        sigma: float,
        dx: float
    ) -> np.ndarray:
        """曲率による圧力ジャンプを計算

        Args:
            levelset: レベルセット場
            spheres: 球体のリスト
            X, Y, Z: 座標グリッド
            sigma: 表面張力係数
            dx: グリッド間隔

        Returns:
            圧力ジャンプの場
        """
        pressure_jump = np.zeros_like(levelset)
        interface_width = 2 * dx  # 界面の幅パラメータ

        # 各球体の界面での圧力ジャンプを計算
        for sphere in spheres:
            # 球からの距離を計算
            distance = np.sqrt(
                (X - sphere.center[0])**2 +
                (Y - sphere.center[1])**2 +
                (Z - sphere.center[2])**2
            )
            
            # 界面近傍の判定
            interface_region = np.abs(distance - sphere.radius) < interface_width
            
            # 球の曲率は半径の逆数（符号は球の内外で反転）
            # 2/Rは平均曲率（主曲率1/Rが2回足される）
            curvature = 2.0 / sphere.radius
            
            # 圧力ジャンプの追加（水球と空気球で符号が逆）
            if sphere.is_water:
                pressure_jump += sigma * curvature * interface_region
            else:
                pressure_jump -= sigma * curvature * interface_region

        # 水面は平面なので曲率による圧力ジャンプなし

        return pressure_jump

    def _process_sphere_config(
        self, obj: Dict[str, Any], domain_size: List[float]
    ) -> SphereConfig:
        """球体の設定を処理

        Args:
            obj: 球体の設定辞書
            domain_size: 領域のサイズ

        Returns:
            処理された球体の設定
        """
        center = obj.get("center", [0.5, 0.5, 0.4])
        radius = obj.get("radius", 0.2)
        phase = obj.get("phase", "nitrogen")

        # 物理的な寸法に変換
        center_phys = [c * d for c, d in zip(center, domain_size)]
        radius_phys = radius * domain_size[0]  # スケーリングはx方向のサイズを使用

        return SphereConfig(
            center=center_phys,
            radius=radius_phys,
            is_water=(phase != "nitrogen")
        )

    def _compute_pressure_field(
        self,
        phase_indicator: np.ndarray,
        domain_size: List[float],
        water_height: float,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        spheres: List[SphereConfig],
        levelset: np.ndarray,
    ) -> np.ndarray:
        """圧力場を計算

        Args:
            phase_indicator: 相インジケータ（True=水, False=空気）
            domain_size: 領域サイズ
            water_height: 水面の高さ
            X, Y, Z: 座標グリッド
            spheres: 球体のリスト
            levelset: レベルセット場

        Returns:
            圧力場
        """
        # 物性値の取得
        water_props = self.fluid_properties.get("water")
        air_props = self.fluid_properties.get("nitrogen")
        rho_w = water_props.density
        rho_a = air_props.density
        sigma = water_props.surface_tension
        g = self.config.get("physics", {}).get("gravity", 9.81)

        # 圧力場の初期化
        pressure = np.zeros_like(X)

        # 基本の静水圧分布を設定
        density = np.where(phase_indicator, rho_w, rho_a)
        pressure = np.where(
            Z > water_height,
            rho_a * g * (domain_size[2] - Z),  # 空気領域
            rho_a * g * (domain_size[2] - water_height) + rho_w * g * (water_height - Z)  # 水領域
        )

        # グリッド間隔
        dx = domain_size[0] / (X.shape[0] - 1)

        # 表面張力による圧力ジャンプを追加
        pressure_jump = self._compute_curvature_and_pressure_jump(
            levelset, spheres, X, Y, Z, sigma, dx
        )
        pressure += pressure_jump

        return pressure

    def _setup_initial_fields(
        self, dimensions: List[int], domain_size: List[float], initial_conditions: Dict
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """初期フィールドを設定

        Args:
            dimensions: グリッドの次元
            domain_size: 領域のサイズ
            initial_conditions: 初期条件の設定

        Returns:
            levelset, pressure, velocityのタプル
        """
        # 座標グリッドの生成
        x = np.linspace(0, domain_size[0], dimensions[0])
        y = np.linspace(0, domain_size[1], dimensions[1])
        z = np.linspace(0, domain_size[2], dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # 背景相の設定
        background = initial_conditions.get("background", {})
        height_fraction = background.get("height_fraction", 0.8)
        water_height = height_fraction * domain_size[2]  # Z方向のサイズを使用

        # 球体の設定を処理
        spheres = []
        for obj in initial_conditions.get("objects", []):
            if obj["type"] == "sphere":
                sphere = self._process_sphere_config(obj, domain_size)
                spheres.append(sphere)

        # レベルセット関数と相インジケータの計算
        levelset, phase_indicator = self._compute_closest_interface_distance(
            Z, water_height, spheres, X, Y
        )

        # 圧力場の計算
        pressure = self._compute_pressure_field(
            phase_indicator,
            domain_size,
            water_height,
            X,
            Y,
            Z,
            spheres,
            levelset,
        )

        # 速度場の初期化（ゼロ）
        velocity = [np.zeros_like(X) for _ in range(3)]

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