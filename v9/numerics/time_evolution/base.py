"""
時間発展ソルバーの抽象基底クラス

数値時間発展計算のための抽象インターフェースを定義します。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar, List, Dict, Any, Optional
from core.field import ScalarField, VectorField


# 汎用的な型変数を定義
FieldType = TypeVar("FieldType", ScalarField, VectorField)


@dataclass
class TimeIntegratorConfig:
    """時間積分器の設定"""

    cfl: float = 0.5
    min_dt: float = 1.0e-6
    max_dt: float = 1.0
    tolerance: float = 1.0e-6
    stability_limit: float = float("inf")


class TimeIntegrator(ABC, Generic[FieldType]):
    """
    時間発展ソルバーの抽象基底クラス

    汎用的な時間積分のためのテンプレートメソッドパターンを提供
    """

    def __init__(self, config: TimeIntegratorConfig = TimeIntegratorConfig()):
        """
        時間積分器を初期化

        Args:
            config: 時間積分の設定パラメータ
        """
        # 設定パラメータの検証
        self._validate_config(config)

        # 設定の保存
        self.config = config

        # 状態追跡用のプロパティ
        self._time: float = 0.0
        self._step_count: int = 0
        self._error_history: List[float] = []

    def _validate_config(self, config: TimeIntegratorConfig):
        """
        設定パラメータの妥当性を検証

        Args:
            config: 検証する設定パラメータ

        Raises:
            ValueError: 不正な設定が見つかった場合
        """
        if not 0 < config.cfl <= 1:
            raise ValueError("CFLは0から1の間である必要があります")

        if config.min_dt <= 0 or config.max_dt <= 0:
            raise ValueError("時間刻み幅は正の値である必要があります")

        if config.min_dt > config.max_dt:
            raise ValueError("最小時間刻み幅は最大時間刻み幅以下である必要があります")

    @abstractmethod
    def integrate(
        self, field: FieldType, dt: float, derivative: FieldType
    ) -> FieldType:
        """
        時間積分を実行する抽象メソッド

        Args:
            field: 現在の場
            dt: 時間刻み幅
            derivative: 場の時間微分

        Returns:
            更新された場
        """
        pass

    @abstractmethod
    def compute_timestep(self, field: FieldType, **kwargs) -> float:
        """
        安定な時間刻み幅を計算する抽象メソッド

        Args:
            field: 現在の場
            **kwargs: 追加のパラメータ

        Returns:
            計算された時間刻み幅
        """
        pass

    def step_forward(
        self, field: FieldType, derivative: FieldType, dt: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        1ステップの時間発展を実行

        Args:
            field: 現在の場
            derivative: 場の時間微分
            dt: 時間刻み幅（Noneの場合は自動計算）

        Returns:
            診断情報の辞書
        """
        # 時間刻み幅の決定
        if dt is None:
            dt = self.compute_timestep(field)

        # 時間発展の実行
        new_field = self.integrate(field, dt, derivative)

        # 状態の更新
        self._time += dt
        self._step_count += 1

        # 診断情報の収集
        return self._collect_diagnostics(new_field, dt)

    def _collect_diagnostics(self, new_field: FieldType, dt: float) -> Dict[str, Any]:
        """
        診断情報を収集

        Args:
            new_field: 更新された場
            dt: 時間刻み幅

        Returns:
            診断情報の辞書
        """
        return {
            "time": self._time,
            "dt": dt,
            "step_count": self._step_count,
            "field_stats": {
                "min": new_field.min() if hasattr(new_field, "min") else None,
                "max": new_field.max() if hasattr(new_field, "max") else None,
                "mean": new_field.mean() if hasattr(new_field, "mean") else None,
            },
            "method": self.__class__.__name__,
        }

    def reset(self):
        """
        積分器の状態をリセット
        """
        self._time = 0.0
        self._step_count = 0
        self._error_history.clear()
