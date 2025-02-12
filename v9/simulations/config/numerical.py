"""数値計算の設定を管理するモジュール"""

from dataclasses import dataclass
from typing import Dict, Any
from .base import BaseConfig, load_config_safely


@dataclass
class NumericalConfig(BaseConfig):
    """数値計算の設定を保持するクラス"""

    time_integrator: str = "euler"
    max_time: float = 2.0
    initial_dt: float = 0.001
    save_interval: float = 0.01
    cfl: float = 0.5
    min_dt: float = 1.0e-6
    max_dt: float = 1.0
    level_set_epsilon: float = 1.0e-2
    level_set_reinit_interval: int = 10  # より適切なデフォルト値に更新
    level_set_reinit_steps: int = 2

    def validate(self) -> None:
        """数値設定の妥当性を検証"""
        # 時間積分器のバリデーション
        if self.time_integrator not in ["euler", "rk4"]:
            raise ValueError("time_integratorはeulerまたはrk4である必要があります")

        # 時間関連パラメータのバリデーション
        if self.max_time <= 0:
            raise ValueError("max_timeは正の値である必要があります")
        if self.initial_dt <= 0:
            raise ValueError("initial_dtは正の値である必要があります")
        if self.save_interval <= 0:
            raise ValueError("save_intervalは正の値である必要があります")
        if not 0 < self.cfl <= 1:
            raise ValueError("cflは0から1の間である必要があります")
        if self.min_dt <= 0:
            raise ValueError("min_dtは正の値である必要があります")
        if self.max_dt <= self.min_dt:
            raise ValueError("max_dtはmin_dtより大きい必要があります")

        # Level Set関連パラメータのバリデーション
        if self.level_set_epsilon <= 0:
            raise ValueError("level_set_epsilonは正の値である必要があります")
        if self.level_set_reinit_interval <= 0:
            raise ValueError("level_set_reinit_intervalは正の値である必要があります")
        if self.level_set_reinit_steps <= 0:
            raise ValueError("level_set_reinit_stepsは正の値である必要があります")

    def load(self, config_dict: Dict[str, Any]) -> "NumericalConfig":
        """辞書から設定を読み込む"""
        # デフォルト値を設定しつつ、入力された値で上書き
        merged_config = load_config_safely(
            config_dict,
            {
                "time_integrator": "euler",
                "max_time": 2.0,
                "initial_dt": 0.001,
                "save_interval": 0.01,
                "cfl": 0.5,
                "min_dt": 1.0e-6,
                "max_dt": 1.0,
                "level_set": {
                    "epsilon": 1.0e-2,
                    "reinit_interval": 10,  # デフォルト値を更新
                    "reinit_steps": 2,
                },
            },
        )

        # Level Set関連パラメータの特別な処理
        level_set_config = merged_config.get("level_set", {})

        return NumericalConfig(
            time_integrator=merged_config.get("time_integrator", "euler"),
            max_time=merged_config.get("max_time", 2.0),
            initial_dt=merged_config.get("initial_dt", 0.001),
            save_interval=merged_config.get("save_interval", 0.01),
            cfl=merged_config.get("cfl", 0.5),
            min_dt=merged_config.get("min_dt", 1.0e-6),
            max_dt=merged_config.get("max_dt", 1.0),
            level_set_epsilon=level_set_config.get("epsilon", 1.0e-2),
            level_set_reinit_interval=level_set_config.get("reinit_interval", 10),  # デフォルト値を更新
            level_set_reinit_steps=level_set_config.get("reinit_steps", 2),
        )

    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式にシリアライズ"""
        return {
            "time_integrator": self.time_integrator,
            "max_time": self.max_time,
            "initial_dt": self.initial_dt,
            "save_interval": self.save_interval,
            "cfl": self.cfl,
            "min_dt": self.min_dt,
            "max_dt": self.max_dt,
            "level_set": {
                "epsilon": self.level_set_epsilon,
                "reinit_interval": self.level_set_reinit_interval,
                "reinit_steps": self.level_set_reinit_steps,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NumericalConfig":
        """辞書から設定を復元"""
        return cls().load(config_dict)