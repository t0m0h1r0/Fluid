"""シミュレーションの初期化を管理するモジュール

このモジュールは、流体シミュレーションの初期状態を設定するためのクラスを提供します。
"""

from typing import Dict, Any, Tuple
import numpy as np

from .state import SimulationState
from physics.properties import FluidProperties


class SimulationInitializer:
    """シミュレーション初期化クラス"""

    def __init__(self, config: Dict[str, Any], logger=None):
        """初期化クラスを初期化

        Args:
            config: 設定辞書
            logger: ロガーオブジェクト
        """
        self.config = config
        self.logger = logger

        # 領域の設定を取得
        self.domain_config = config.get("domain", {})
        self.nx = self.domain_config.get("dimensions", [64, 64, 64])[0]
        self.ny = self.domain_config.get("dimensions", [64, 64, 64])[1]
        self.nz = self.domain_config.get("dimensions", [64, 64, 64])[2]
        self.shape = (self.nx, self.ny, self.nz)

        # 物理領域のサイズを取得
        self.lx = self.domain_config.get("size", [1.0, 1.0, 1.0])[0]
        self.ly = self.domain_config.get("size", [1.0, 1.0, 1.0])[1]
        self.lz = self.domain_config.get("size", [1.0, 1.0, 1.0])[2]

        # グリッド間隔を計算
        self.dx = self.lx / self.nx

        # 初期条件の設定を取得
        self.initial_config = config.get("initial_conditions", {})

        # 物性値の設定を取得
        self.physics_config = config.get("physics", {})

    def _create_phase_properties(self) -> Tuple[FluidProperties, FluidProperties]:
        """相の物性値を作成

        Returns:
            phase1, phase2の物性値オブジェクト
        """
        phases_config = self.physics_config.get("phases", {})

        # デフォルト値
        water_config = phases_config.get("water", {})
        water = FluidProperties(
            density=water_config.get("density", 1000.0),
            viscosity=water_config.get("viscosity", 1.0e-3),
            surface_tension=water_config.get("surface_tension", 0.07),
        )

        nitrogen_config = phases_config.get("nitrogen", {})
        nitrogen = FluidProperties(
            density=nitrogen_config.get("density", 1.25),
            viscosity=nitrogen_config.get("viscosity", 1.81e-5),
            surface_tension=nitrogen_config.get("surface_tension", 0.0),
        )

        return water, nitrogen

    def _initialize_levelset(self) -> np.ndarray:
        """Level Set関数を初期化

        Returns:
            初期化されたLevel Set関数の値
        """
        background = self.initial_config.get("background", {})
        height_fraction = background.get("height_fraction", 0.8)
        height = height_fraction * self.lz

        # 格子点の座標を生成
        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        z = np.linspace(0, self.lz, self.nz)

        # meshgridを使用して3D配列を作成
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        phi = height - Z  # 高さに基づいて初期化

        # 物体の追加
        objects = self.initial_config.get("objects", [])
        for obj in objects:
            if obj["type"] == "sphere":
                center = obj["center"]
                radius = obj["radius"]
                x = np.linspace(0, self.lx, self.nx)
                y = np.linspace(0, self.ly, self.ny)
                z = np.linspace(0, self.lz, self.nz)
                X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
                distance = np.sqrt(
                    (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
                )
                phi = np.minimum(phi, radius - distance)

        return phi

    def _initialize_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """速度場を初期化

        Returns:
            x, y, z方向の速度成分
        """
        velocity_config = self.initial_config.get("velocity", {})
        velocity_type = velocity_config.get("type", "zero")

        if velocity_type == "zero":
            return (
                np.zeros(self.shape),
                np.zeros(self.shape),
                np.zeros(self.shape),
            )
        else:
            raise ValueError(f"未対応の初期速度タイプ: {velocity_type}")

    def create_initial_state(self) -> SimulationState:
        """初期状態を生成

        Returns:
            初期化されたシミュレーション状態
        """
        if self.logger:
            self.logger.info("初期状態を生成中...")

        # シミュレーション状態の作成
        state = SimulationState(shape=self.shape, dx=self.dx)

        try:
            # Level Set場の初期化
            state.levelset.data = self._initialize_levelset()

            # 速度場の初期化
            vx, vy, vz = self._initialize_velocity()
            state.velocity.components[0].data = vx
            state.velocity.components[1].data = vy
            state.velocity.components[2].data = vz

            # 圧力場の初期化（デフォルトでゼロ）

            if self.logger:
                self.logger.info("初期状態の生成が完了しました")

        except Exception as e:
            if self.logger:
                self.logger.error(f"初期状態の生成中にエラーが発生: {e}")
            raise

        return state
