"""シミュレーションの初期化を担当するモジュール

リファクタリングされたphysics/パッケージに対応した更新版
"""

from typing import Optional, Tuple
import numpy as np

from physics.levelset import LevelSetField, LevelSetParameters
from physics.levelset.properties import LevelSetPropertiesManager

from core.field import VectorField, ScalarField

from .config import SimulationConfig, ObjectConfig
from .state import SimulationState


class SimulationInitializer:
    """二相流シミュレーションの初期化クラス"""

    def __init__(
        self, 
        config: SimulationConfig, 
        properties: Optional[LevelSetPropertiesManager] = None
    ):
        """初期化クラスを初期化

        Args:
            config: シミュレーション設定
            properties: 物性値マネージャー（オプション）
        """
        self.config = config
        self.properties = properties or LevelSetPropertiesManager(
            phase1=config.phases["water"].to_properties(),
            phase2=config.phases["nitrogen"].to_properties()
        )

    def create_initial_state(self) -> SimulationState:
        """初期状態を生成

        Returns:
            初期化された状態
        """
        # グリッドの設定
        shape = tuple(self.config.domain.dimensions)
        dx = self.config.domain.size[0] / shape[0]

        # フィールドの初期化
        velocity = self._initialize_velocity(shape, dx)
        levelset = self._initialize_levelset(shape, dx)
        pressure = self._initialize_pressure(shape, dx)

        # 状態の作成
        return SimulationState(
            velocity=velocity,
            levelset=levelset,
            pressure=pressure,
            properties=self.properties
        )

    def _initialize_velocity(
        self, 
        shape: Tuple[int, ...], 
        dx: float
    ) -> VectorField:
        """速度場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔

        Returns:
            初期化された速度場
        """
        velocity = VectorField(shape, dx)
        
        # 初期速度設定
        vel_config = self.config.initial_condition.velocity
        if vel_config.get("type", "zero") == "uniform":
            values = vel_config.get("values", [0, 0, 0])
            for i, component in enumerate(velocity.components):
                if i < len(values):
                    component.data.fill(values[i])

        return velocity

    def _initialize_levelset(
        self, 
        shape: Tuple[int, ...], 
        dx: float
    ) -> LevelSetField:
        """レベルセット場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔

        Returns:
            初期化されたレベルセット場
        """
        levelset = LevelSetField(
            shape, 
            dx, 
            params=LevelSetParameters(**self.config.solver.level_set)
        )

        # 背景水層の初期化
        if self.config.initial_condition.background_layer:
            height = self.config.initial_condition.background_layer
            z = np.linspace(0, 1, shape[2])
            Z = np.tile(z, (shape[0], shape[1], 1))
            levelset.data = height - Z

        # オブジェクトの初期化
        for obj in self.config.initial_condition.objects:
            self._initialize_object(levelset, obj)

        return levelset

    def _initialize_object(
        self, 
        levelset: LevelSetField, 
        obj: ObjectConfig
    ):
        """オブジェクトを初期化

        Args:
            levelset: レベルセット場
            obj: オブジェクトの設定
        """
        if obj.type == "sphere":
            center = np.array(obj.center)
            radius = obj.radius

            # グリッド座標の生成
            x = np.linspace(0, 1, levelset.shape[0])
            y = np.linspace(0, 1, levelset.shape[1])
            z = np.linspace(0, 1, levelset.shape[2])
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

            # 球の距離関数を計算
            dist = np.sqrt(
                (X - center[0])**2 + 
                (Y - center[1])**2 + 
                (Z - center[2])**2
            )

            # レベルセット場を更新
            phi_sphere = dist - radius
            phi_sphere *= -1 if obj.phase == "water" else 1
            
            if obj.phase == "water":
                levelset.data = np.minimum(levelset.data, phi_sphere)
            else:
                levelset.data = np.maximum(levelset.data, phi_sphere)

    def _initialize_pressure(
        self, 
        shape: Tuple[int, ...], 
        dx: float
    ) -> ScalarField:
        """静水圧分布を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔

        Returns:
            初期化された圧力場
        """
        pressure = ScalarField(shape, dx)

        # 重力加速度
        g = self.config.physics.gravity

        # 密度場の計算
        density = self.properties.compute_density(
            np.zeros(shape)  # 初期状態での密度計算用の疑似レベルセット
        )

        # 高さ方向に静水圧分布を設定
        z = np.linspace(0, 1, shape[2])
        Z = np.tile(z, (shape[0], shape[1], 1))
        
        pressure.data = density * g * (1.0 - Z)

        return pressure