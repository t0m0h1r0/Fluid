"""シミュレーションの初期化を担当するモジュール"""

from .config import SimulationConfig
from .state import SimulationState
from physics.levelset import LevelSetField
from physics.levelset.initializer import LevelSetInitializer, InterfaceObject
from core.field import VectorField, ScalarField


class SimulationInitializer:
    """シミュレーションの初期化を担当するクラス"""

    def __init__(self, config: SimulationConfig):
        """初期化子を構築

        Args:
            config: シミュレーション設定
        """
        self.config = config
        self.validate_config()

    def validate_config(self) -> None:
        """設定の妥当性を検証"""
        self.config.validate()

    def create_initial_state(self) -> SimulationState:
        """初期状態を生成

        Returns:
            初期化されたシミュレーション状態
        """
        # グリッドの形状とスケーリングを計算
        shape = tuple(self.config.domain.dimensions)
        domain_size = self.config.domain.size
        dx = [size / dim for size, dim in zip(domain_size, shape)]

        # グリッド間隔は最小値を使用（CFL条件のため）
        min_dx = min(dx)

        # 速度場を初期化（ゼロで初期化）
        velocity = VectorField(shape, dx=min_dx)

        # レベルセット場を初期化
        levelset = self._initialize_levelset(shape, domain_size, min_dx)

        # 圧力場を初期化（ゼロで初期化）
        pressure = ScalarField(shape, dx=min_dx)

        return SimulationState(
            time=0.0, velocity=velocity, levelset=levelset, pressure=pressure
        )

    def _initialize_levelset(
        self, shape: tuple, domain_size: list, dx: float
    ) -> LevelSetField:
        """レベルセット場を初期化

        Args:
            shape: グリッドの形状
            domain_size: 計算領域の物理サイズ
            dx: グリッド間隔

        Returns:
            初期化されたLevel Set場
        """
        # インターフェースオブジェクトの作成（物理スケールに変換）
        interface_objects = []

        # 背景相の追加
        background_config = self.config.initial_conditions.background
        interface_objects.append(
            InterfaceObject(phase=background_config["phase"], object_type="background")
        )

        # その他のオブジェクトの追加
        for obj in self.config.initial_conditions.get("objects", []):
            scaled_obj = self._scale_interface_object(obj, domain_size)
            interface_objects.append(scaled_obj)

        # Level Set初期化子を使用
        initializer = LevelSetInitializer(
            dx=dx, background_phase=background_config["phase"]
        )
        return initializer.initialize(shape=shape, objects=interface_objects)

    def _scale_interface_object(self, obj: dict, domain_size: list) -> InterfaceObject:
        """インターフェースオブジェクトを物理スケールに変換

        Args:
            obj: オブジェクトの設定辞書
            domain_size: 計算領域の物理サイズ

        Returns:
            スケーリングされたインターフェースオブジェクト
        """
        object_type = obj["type"]
        phase = obj["phase"]

        if object_type == "layer":
            # 高さは相対値（0-1）のまま
            return InterfaceObject(
                phase=phase, object_type=object_type, height=obj["height"]
            )

        elif object_type == "sphere":
            # 中心座標を物理座標に変換
            physical_center = [c * size for c, size in zip(obj["center"], domain_size)]
            # 半径を物理スケールに変換（最小領域サイズに対する比率を保持）
            physical_radius = obj["radius"] * min(domain_size)

            return InterfaceObject(
                phase=phase,
                object_type=object_type,
                center=physical_center,
                radius=physical_radius,
            )

        else:
            raise ValueError(f"未対応のオブジェクトタイプ: {object_type}")
