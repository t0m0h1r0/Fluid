"""シミュレーションの初期化を担当するモジュール

このモジュールは、二相流シミュレーションの初期条件を設定します。
速度場、レベルセット場、圧力場の初期化を行います。
"""

from typing import Dict, Any, List
import numpy as np

from physics.levelset import LevelSetField, LevelSetParameters
from physics.levelset.properties import PropertiesManager
from core.field import VectorField, ScalarField
from simulations.config.simulation_config import SimulationConfig, ObjectConfig
from simulations.state import SimulationState


class TwoPhaseFlowInitializer:
    """二相流シミュレーションの初期化クラス"""

    def __init__(
        self, config: SimulationConfig, properties: PropertiesManager, logger=None
    ):
        """初期化クラスを初期化

        Args:
            config: シミュレーション設定
            properties: 物性値マネージャー
            logger: ロガー
        """
        self.config = config
        self.properties = properties
        self.logger = logger

    def create_initial_state(self) -> SimulationState:
        """初期状態を生成

        Returns:
            初期化された状態
        """
        try:
            # グリッドの設定
            shape = tuple(self.config.domain.dimensions)
            dx = self.config.domain.size[0] / shape[0]

            # フィールドの作成
            velocity = VectorField(shape, dx)
            levelset = LevelSetField(
                shape, dx, params=LevelSetParameters(**self.config.solver.level_set)
            )
            pressure = ScalarField(shape, dx)

            # 初期条件の適用
            self._apply_initial_conditions(velocity, levelset, pressure)

            # 状態の作成
            state = SimulationState(
                velocity=velocity,
                levelset=levelset,
                pressure=pressure,
                properties=self.properties,
            )

            if self.logger:
                self.logger.info(
                    f"初期状態を生成しました\n"
                    f"  レベルセット体積: {levelset.compute_volume():.6g}\n"
                    f"  圧力範囲: [{np.min(pressure.data):.6g}, "
                    f"{np.max(pressure.data):.6g}]"
                )

            return state

        except Exception as e:
            if self.logger:
                self.logger.error(f"初期状態の生成中にエラー: {e}")
            raise

    def _apply_initial_conditions(
        self, velocity: VectorField, levelset: LevelSetField, pressure: ScalarField
    ):
        """初期条件を適用

        Args:
            velocity: 速度場
            levelset: レベルセット場
            pressure: 圧力場
        """
        # 背景の水層を初期化
        if self.config.initial_condition.background_layer:
            self._initialize_water_layer(
                levelset, self.config.initial_condition.background_layer
            )

        # オブジェクトの初期化
        for obj in self.config.initial_condition.objects:
            self._initialize_object(levelset, obj)

        # 静水圧分布の計算
        self._initialize_hydrostatic_pressure(pressure, levelset)

        # 初期速度場の設定
        if self.config.initial_condition.velocity:
            self._initialize_velocity(velocity)

    def _initialize_water_layer(self, levelset: LevelSetField, height_fraction: float):
        """水層を初期化

        Args:
            levelset: レベルセット場
            height_fraction: 水面の高さ（無次元）
        """
        # 座標グリッドの生成
        z = np.linspace(0, 1, levelset.shape[2])
        Z = np.tile(z, (levelset.shape[0], levelset.shape[1], 1))

        # レベルセット場の初期化（水面からの符号付き距離）
        height = height_fraction
        levelset.data = height - Z

    def _initialize_object(self, levelset: LevelSetField, obj: ObjectConfig):
        """オブジェクトを初期化

        Args:
            levelset: レベルセット場
            obj: オブジェクトの設定
        """
        if obj.type == "sphere":
            self._initialize_sphere(
                levelset, obj.center, obj.radius, obj.phase == "water"
            )
        else:
            if self.logger:
                self.logger.warning(f"未対応のオブジェクト種別: {obj.type}")

    def _initialize_sphere(
        self,
        levelset: LevelSetField,
        center: List[float],
        radius: float,
        is_water: bool,
    ):
        """球を初期化

        Args:
            levelset: レベルセット場
            center: 球の中心座標（無次元）
            radius: 球の半径（無次元）
            is_water: 水球かどうか
        """
        # 座標グリッドの生成
        x = np.linspace(0, 1, levelset.shape[0])
        y = np.linspace(0, 1, levelset.shape[1])
        z = np.linspace(0, 1, levelset.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # 球からの符号付き距離を計算
        distance = np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )

        # 現在のレベルセット場と合成（最小値を取る）
        phi_sphere = (-1 if is_water else 1) * (distance - radius)
        if is_water:
            levelset.data = np.minimum(levelset.data, phi_sphere)
        else:
            levelset.data = np.maximum(levelset.data, phi_sphere)

    def _initialize_hydrostatic_pressure(
        self, pressure: ScalarField, levelset: LevelSetField
    ):
        """静水圧分布を初期化

        Args:
            pressure: 圧力場
            levelset: レベルセット場
        """
        # 密度場の取得
        density = self.properties.get_density(levelset)

        # 重力加速度
        g = self.config.physics.gravity

        # 静水圧分布の計算
        z = np.linspace(0, 1, pressure.shape[2])
        Z = np.tile(z, (pressure.shape[0], pressure.shape[1], 1))
        pressure.data = density * g * (1.0 - Z)

    def _initialize_velocity(self, velocity: VectorField):
        """速度場を初期化

        Args:
            velocity: 速度場
        """
        vel_config = self.config.initial_condition.velocity

        # vel_configが文字列の場合（"zero"など）の対応
        if isinstance(vel_config, str):
            vel_config = {"type": vel_config}

        # デフォルトで速度はゼロ
        vel_type = vel_config.get("type", "zero")

        if vel_type == "zero":
            # ゼロ初期化（デフォルト）
            pass
        elif vel_type == "uniform":
            # 一様流れ
            values = vel_config.get("values", [0, 0, 0])
            for i, component in enumerate(velocity.components):
                if i < len(values):
                    component.data.fill(values[i])
        elif vel_type == "vortex":
            # 渦
            center = vel_config.get("center", [0.5, 0.5, 0.5])
            strength = vel_config.get("strength", 1.0)

            # 座標グリッドの生成
            x = np.linspace(0, 1, velocity.shape[0])
            y = np.linspace(0, 1, velocity.shape[1])
            z = np.linspace(0, 1, velocity.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            # 中心からの距離を計算
            R = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            R = np.maximum(R, velocity.dx)  # ゼロ除算を防ぐ

            # 渦の速度成分を計算
            velocity.components[0].data = -strength * (Y - center[1]) / R
            velocity.components[1].data = strength * (X - center[0]) / R
            # Z方向の速度はゼロのまま
        else:
            if self.logger:
                self.logger.warning(f"未対応の速度場初期化タイプ: {vel_type}")

    def _initialize_vortex(self, velocity: VectorField, vel_config: Dict[str, Any]):
        """渦を初期化

        Args:
            velocity: 速度場
            vel_config: 渦の設定
        """
        # 渦の中心と強さ
        center = vel_config.get("center", [0.5, 0.5, 0.5])
        strength = vel_config.get("strength", 1.0)

        # 座標グリッドの生成
        x = np.linspace(0, 1, velocity.shape[0])
        y = np.linspace(0, 1, velocity.shape[1])
        z = np.linspace(0, 1, velocity.shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # 中心からの距離を計算
        R = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        R = np.maximum(R, velocity.dx)  # ゼロ除算を防ぐ

        # 渦の速度成分を計算
        velocity.components[0].data = -strength * (Y - center[1]) / R
        velocity.components[1].data = strength * (X - center[0]) / R
        # Z方向の速度はゼロのまま
