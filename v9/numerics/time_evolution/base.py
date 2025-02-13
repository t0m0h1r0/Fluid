"""時間発展ソルバーの基底クラスを提供するモジュール

このモジュールは、数値時間発展計算のための基底クラスとインターフェースを定義します。
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
from datetime import datetime
import numpy as np

from core.field import ScalarField, VectorField

# 入力として受け付けるフィールドの型
FieldType = Union[ScalarField, VectorField]


class TimeIntegrator(ABC):
    """時間積分の基底クラス"""

    def __init__(
        self,
        cfl: float = 0.5,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
        tolerance: float = 1e-6,
        stability_limit: float = float("inf"),
    ):
        """時間積分器を初期化

        Args:
            cfl: CFL条件の係数
            min_dt: 最小時間刻み幅
            max_dt: 最大時間刻み幅
            tolerance: 収束判定の許容誤差
            stability_limit: 安定性限界
        """
        self._validate_parameters(cfl, min_dt, max_dt, tolerance)
        self._cfl = cfl
        self._min_dt = min_dt
        self._max_dt = max_dt
        self._tolerance = tolerance
        self._stability_limit = stability_limit
        self._time = 0.0
        self._dt = None
        self._start_time = None
        self._step_count = 0
        self._error_history = []

    def _validate_parameters(
        self, cfl: float, min_dt: float, max_dt: float, tolerance: float
    ) -> None:
        """パラメータの妥当性を検証"""
        if not 0 < cfl <= 1:
            raise ValueError("CFLは0から1の間である必要があります")
        if min_dt <= 0 or max_dt <= 0:
            raise ValueError("時間刻み幅は正である必要があります")
        if min_dt > max_dt:
            raise ValueError("最小時間刻み幅は最大時間刻み幅以下である必要があります")
        if tolerance <= 0:
            raise ValueError("許容誤差は正である必要があります")

    @abstractmethod
    def integrate(
        self,
        field: FieldType,
        dt: float,
        derivative: FieldType,
    ) -> FieldType:
        """時間積分を実行

        Args:
            field: 現在のフィールド値（ScalarFieldまたはVectorField）
            dt: 時間刻み幅
            derivative: フィールドの時間微分（fieldと同じ型）

        Returns:
            更新されたフィールド（fieldと同じ型）
        """
        pass

    def step_forward(
        self,
        field: FieldType,
        derivative: FieldType,
        dt: Optional[float] = None,
    ) -> tuple[FieldType, Dict[str, Any]]:
        """1ステップの時間発展を実行

        Args:
            field: 現在のフィールド値
            derivative: フィールドの時間微分
            dt: 時間刻み幅（Noneの場合は自動計算）

        Returns:
            (更新されたフィールド, 診断情報)のタプル
        """
        if not isinstance(field, (ScalarField, VectorField)):
            raise TypeError("fieldはScalarFieldまたはVectorFieldである必要があります")

        if not isinstance(derivative, type(field)):
            raise TypeError("derivativeはfieldと同じ型である必要があります")

        # 開始時刻の記録
        if self._start_time is None:
            self._start_time = datetime.now()

        # 時間刻み幅の決定
        if dt is None:
            dt = self.compute_timestep(field)
        dt = self._validate_timestep(dt)

        try:
            # 時間発展の実行
            new_field = self.integrate(field, dt, derivative)

            # 時刻の更新
            self._time += dt
            self._dt = dt
            self._step_count += 1

            # 診断情報の収集と返却
            diagnostics = self._create_diagnostics(dt)
            return new_field, diagnostics

        except Exception as e:
            raise RuntimeError(f"時間発展中にエラー: {e}")

    def compute_timestep(self, field: FieldType, **kwargs) -> float:
        """安定な時間刻み幅を計算"""
        return self._max_dt

    def _validate_timestep(self, dt: float) -> float:
        """時間刻み幅の妥当性を検証"""
        if dt <= 0:
            raise ValueError("時間刻み幅は正である必要があります")
        return self._clip_timestep(dt)

    def _clip_timestep(self, dt: float) -> float:
        """時間刻み幅を許容範囲に制限"""
        return np.clip(dt, self._min_dt, self._max_dt)

    def _create_diagnostics(self, dt: float) -> Dict[str, Any]:
        """診断情報を生成"""
        return {
            "time": self._time,
            "dt": dt,
            "method": self.__class__.__name__,
            "order": self.get_order(),
            "step_count": self._step_count,
            "error_estimate": self.get_error_estimate(),
            "elapsed_time": self.elapsed_time,
            "stability_limit": self._stability_limit,
        }

    @property
    def elapsed_time(self) -> Optional[float]:
        """経過時間を計算"""
        if self._start_time is None:
            return None
        return (datetime.now() - self._start_time).total_seconds()

    @abstractmethod
    def get_order(self) -> int:
        """数値スキームの次数を取得"""
        pass

    @abstractmethod
    def get_error_estimate(self) -> float:
        """誤差の推定値を取得"""
        pass

    def reset(self):
        """積分器の状態をリセット"""
        self._time = 0.0
        self._dt = None
        self._start_time = None
        self._step_count = 0
        self._error_history.clear()
