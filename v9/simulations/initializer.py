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
        # グリッドの形状を取得
        shape = tuple(self.config.domain.dimensions)  # .values()を削除
        dx = min(self.config.domain.size)

        # 速度場を初期化（ゼロで初期化）
        velocity = VectorField(shape)

        # レベルセット場を初期化
        levelset = self._initialize_levelset(shape, dx)

        # 圧力場を初期化（ゼロで初期化）
        pressure = ScalarField(shape)

        return SimulationState(
            time=0.0, velocity=velocity, levelset=levelset, pressure=pressure
        )

    def _initialize_levelset(self, shape: tuple, dx: float) -> LevelSetField:
        """レベルセット場を初期化"""
        # インターフェースオブジェクトの作成
        interface_objects = [
            InterfaceObject(
                phase=interface.phase,
                object_type=interface.object_type,
                height=interface.height,
                center=interface.center,
                radius=interface.radius,
            )
            for interface in self.config.interfaces
        ]

        # Level Set初期化子を使用
        initializer = LevelSetInitializer(dx=dx)
        return initializer.initialize(shape=shape, objects=interface_objects)
