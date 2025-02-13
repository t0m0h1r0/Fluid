"""シミュレーションの初期化を担当するモジュール

このモジュールは、二相流シミュレーションの初期状態を設定します。
設定ファイルに基づいて、速度場、圧力場、界面関数などを
適切に初期化します。
"""

import numpy as np

from core.field import VectorField, ScalarField
from physics.multiphase import InterfaceOperations
from .config import SimulationConfig
from .state import SimulationState


class SimulationInitializer:
    """シミュレーション初期化クラス"""

    def __init__(self, config: SimulationConfig):
        """初期化クラスを構築

        Args:
            config: シミュレーション設定
        """
        self.config = config
        self._validate_config()
        
        # InterfaceOperationsの初期化
        self._interface_ops = InterfaceOperations(
            dx=self.config.domain.size[0] / self.config.domain.dimensions[0],
            epsilon=self.config.numerical.get('interface', {}).get('epsilon', 1e-2)
        )

    def _validate_config(self):
        """設定の妥当性を検証"""
        if not self.config.domain:
            raise ValueError("領域設定が存在しません")
        if not self.config.physics:
            raise ValueError("物理設定が存在しません")
        if not self.config.initial_conditions:
            raise ValueError("初期条件が存在しません")

    def create_initial_state(self) -> SimulationState:
        """初期状態を生成

        Returns:
            初期化されたシミュレーション状態
        """
        # グリッドの設定
        shape = tuple(self.config.domain.dimensions)
        dx = self.config.domain.size[0] / shape[0]  # 等方グリッドを仮定

        # 速度場の初期化
        velocity = self._initialize_velocity(shape, dx)

        # 界面関数の初期化
        levelset = self._initialize_interface(shape, dx)

        # 圧力場の初期化
        pressure = ScalarField(shape, dx)

        # 状態の構築
        state = SimulationState(
            time=0.0,
            velocity=velocity,
            levelset=levelset,  # ScalarFieldとして渡す
            pressure=pressure,
        )

        return state

    def _initialize_velocity(self, shape: tuple, dx: float) -> VectorField:
        """速度場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔

        Returns:
            初期化された速度場
        """
        velocity = VectorField(shape, dx)
        velocity_config = self.config.initial_conditions.velocity

        if velocity_config["type"] == "zero":
            # ゼロ速度場（デフォルト）
            pass
        elif velocity_config["type"] == "uniform":
            # 一様流れ
            direction = velocity_config.get("direction", [1.0, 0.0, 0.0])
            magnitude = velocity_config.get("magnitude", 1.0)
            for i, comp in enumerate(velocity.components):
                comp.data.fill(direction[i] * magnitude)
        elif velocity_config["type"] == "vortex":
            # 渦流れ
            center = velocity_config.get("center", [0.5, 0.5, 0.5])
            strength = velocity_config.get("strength", 1.0)
            coords = np.meshgrid(*[np.linspace(0, 1, s) for s in shape], indexing="ij")
            r = np.sqrt(
                sum((c - cent) ** 2 for c, cent in zip(coords[:-1], center[:-1]))
            )
            velocity.components[0].data = -strength * (coords[1] - center[1]) / r
            velocity.components[1].data = strength * (coords[0] - center[0]) / r

        return velocity

    def _initialize_interface(self, shape: tuple, dx: float) -> ScalarField:
        """界面関数を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔

        Returns:
            初期化された界面関数（ScalarField）
        """
        # 界面設定の取得
        initial_conditions = self.config.initial_conditions
        objects = initial_conditions.objects if initial_conditions.objects else []

        # デフォルトの平面界面を生成
        if not objects:
            levelset = self._interface_ops.create_plane(
                shape=shape, 
                normal=[0, 0, 1], 
                point=[0.5, 0.5, 0.5]
            )
        else:
            # 最初のオブジェクトに基づいて界面を生成
            first_obj = objects[0]
            
            if first_obj.get("type") == "plate":
                height = first_obj.get("height", 0.5)
                levelset = self._interface_ops.create_plane(
                    shape=shape, 
                    normal=[0, 0, 1], 
                    point=[0.5, 0.5, height]
                )
            elif first_obj.get("type") == "sphere":
                center = first_obj.get("center", [0.5, 0.5, 0.5])
                radius = first_obj.get("radius", 0.1)
                levelset = self._interface_ops.create_sphere(
                    shape=shape, 
                    center=center, 
                    radius=radius
                )
            else:
                # フォールバック: デフォルトの平面界面
                levelset = self._interface_ops.create_plane(
                    shape=shape, 
                    normal=[0, 0, 1], 
                    point=[0.5, 0.5, 0.5]
                )

        # 残りのオブジェクトを組み合わせる
        for obj in objects[1:]:
            if obj.get("type") == "plate":
                height = obj.get("height", 0.5)
                plate = self._interface_ops.create_plane(
                    shape=shape, 
                    normal=[0, 0, 1], 
                    point=[0.5, 0.5, height]
                )
                levelset = self._interface_ops.combine_interfaces(levelset, plate, "union")
            elif obj.get("type") == "sphere":
                center = obj.get("center", [0.5, 0.5, 0.5])
                radius = obj.get("radius", 0.1)
                sphere = self._interface_ops.create_sphere(
                    shape=shape, 
                    center=center, 
                    radius=radius
                )
                levelset = self._interface_ops.combine_interfaces(levelset, sphere, "union")

        return levelset