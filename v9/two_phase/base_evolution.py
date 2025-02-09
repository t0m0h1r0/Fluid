from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseEvolution(ABC):
    """数値シミュレーションにおける時間発展の基底クラス"""

    def __init__(
        self,
        name: str,
        time_integrator: str = "rk4",
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
    ):
        """基底進化クラスを初期化

        Args:
            name: 進化クラスの名前
            time_integrator: 時間積分スキーム
            cfl: CFL数
            min_dt: 最小時間ステップ
            max_dt: 最大時間ステップ
        """
        self.name = name
        self.time_integrator = time_integrator
        self.cfl = cfl
        self.min_dt = min_dt
        self.max_dt = max_dt

    @abstractmethod
    def compute_timestep(self, current_state, **kwargs) -> float:
        """時間ステップを計算

        Args:
            current_state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間ステップ
        """
        pass

    @abstractmethod
    def advance(self, current_state, dt: float, **kwargs) -> Dict[str, Any]:
        """状態を時間発展

        Args:
            current_state: 現在の状態
            dt: 時間ステップ
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態と診断情報の辞書
        """
        pass

    def _validate_time_integrator(self, time_integrator: str):
        """時間積分スキームを検証

        Args:
            time_integrator: 時間積分スキーム名

        Raises:
            ValueError: サポートされていない時間積分スキームの場合
        """
        supported_integrators = ["euler", "rk4"]
        if time_integrator not in supported_integrators:
            raise ValueError(
                f"サポートされていない時間積分スキーム: {time_integrator}. "
                f"サポートされているスキーム: {supported_integrators}"
            )
