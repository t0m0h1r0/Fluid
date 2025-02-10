"""チェックポイント管理を提供するモジュール"""

from pathlib import Path
import numpy as np

from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField
from physics.levelset.properties import PropertiesManager
from .config import SimulationConfig
from .state import SimulationState


class CheckpointManager:
    """チェックポイントの保存と読み込みを管理"""

    def __init__(self, config: SimulationConfig):
        """チェックポイントマネージャーを初期化"""
        self.config = config
        self.checkpoint_dir = Path(config.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: SimulationState, name: str = None):
        """チェックポイントを保存

        Args:
            state: シミュレーション状態
            name: チェックポイント名（省略時は時刻を使用）
        """
        # ファイル名の生成
        if name is None:
            name = f"checkpoint_{state.time:.3f}"
        filepath = self.checkpoint_dir / f"{name}.npz"

        # 状態の保存
        checkpoint_data = {
            "velocity": state.velocity.save_state(),
            "levelset": state.levelset.save_state(),
            "pressure": state.pressure.save_state(),
            "time": state.time,
        }

        np.savez(filepath, **checkpoint_data)

    def load(
        self,
        config: SimulationConfig,
        properties: PropertiesManager,
        logger=None,
        name: str = None,
    ) -> SimulationState:
        """チェックポイントから読み込み

        Args:
            config: シミュレーション設定
            properties: 物性値マネージャー
            logger: ロガー
            name: チェックポイント名（省略時は最新）

        Returns:
            読み込まれたシミュレーション状態
        """
        # チェックポイントファイルの選択
        if name is None:
            # 最新のチェックポイントを探す
            checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.npz"))
            if not checkpoints:
                raise FileNotFoundError("チェックポイントが見つかりません")
            filepath = max(checkpoints, key=lambda p: float(p.stem.split("_")[1]))
        else:
            filepath = self.checkpoint_dir / f"{name}.npz"

        # チェックポイントの読み込み
        checkpoint = np.load(filepath)

        # グリッドの設定
        shape = config.domain.dimensions
        dx = config.domain.size[0] / shape[0]

        # フィールドの復元
        velocity = VectorField(shape, dx)
        velocity.load_state(checkpoint["velocity"].item())

        levelset = LevelSetField(shape, dx)
        levelset.load_state(checkpoint["levelset"].item())

        pressure = ScalarField(shape, dx)
        pressure.load_state(checkpoint["pressure"].item())

        # シミュレーション状態の再構築
        return SimulationState(
            velocity=velocity,
            levelset=levelset,
            pressure=pressure,
            time=float(checkpoint["time"]),
            properties=properties,
        )
