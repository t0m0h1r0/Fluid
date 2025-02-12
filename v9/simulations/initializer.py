"""シミュレーションの初期化を担当するモジュール"""

from typing import Dict, Any
from .config import SimulationConfig
from .state import SimulationState
from physics.levelset import LevelSetField
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
        dx = [size / (dim - 1) for size, dim in zip(domain_size, shape)]

        # グリッド間隔は最小値を使用（CFL条件のため）
        min_dx = min(dx)

        # 速度場を初期化（ゼロで初期化）
        velocity = VectorField(shape, dx=min_dx)

        # レベルセット場を初期化
        levelset = self._initialize_levelset(shape, min_dx)

        # 圧力場を初期化（ゼロで初期化）
        pressure = ScalarField(shape, dx=min_dx)

        return SimulationState(
            time=0.0, velocity=velocity, levelset=levelset, pressure=pressure
        )

    def _initialize_levelset(self, shape: tuple, dx: float) -> LevelSetField:
        """レベルセット場を初期化

        Args:
            shape: グリッドの形状
            dx: グリッド間隔

        Returns:
            初期化されたLevel Set場
        """
        # レベルセット場のインスタンス化
        levelset = LevelSetField(shape=shape, dx=dx)

        # インターフェース設定の取得
        background = self.config.initial_conditions.background
        objects = self.config.initial_conditions.get("objects", [])

        # 初期化パラメータの構築
        init_params = {"background_phase": background["phase"], "objects": []}

        # オブジェクトの処理
        for obj in objects:
            init_obj = self._prepare_interface_object(obj)
            if init_obj:
                init_params["objects"].append(init_obj)

        # レベルセット場の初期化
        levelset.initialize(**init_params)

        return levelset

    def _prepare_interface_object(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        """インターフェースオブジェクトの初期化パラメータを準備

        Args:
            obj: オブジェクトの設定辞書

        Returns:
            初期化用のパラメータ辞書
        """
        object_type = obj["type"]
        phase = obj["phase"]
        domain_size = self.config.domain.size

        if object_type == "layer":
            return {
                "method": "plane",
                "phase": phase,
                "normal": [0, 0, 1],
                "point": [0, 0, obj["height"] * domain_size[2]],
            }

        elif object_type == "sphere":
            # 中心座標を物理座標に変換
            center = [c * size for c, size in zip(obj["center"], domain_size)]
            # 半径を物理スケールに変換
            radius = obj["radius"] * min(domain_size)

            return {
                "method": "sphere",
                "phase": phase,
                "center": center,
                "radius": radius,
            }

        else:
            raise ValueError(f"未対応のオブジェクトタイプ: {object_type}")
