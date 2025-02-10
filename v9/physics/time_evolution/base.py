"""時間積分スキームを提供するモジュール

このモジュールは、微分方程式の時間積分を行うための数値スキームを提供します。
前進Euler法やRunge-Kutta法など、様々な時間積分手法を実装します。
"""

from typing import Any, Callable, Dict, Protocol


class TimeIntegratorBase(Protocol):
    """時間積分のプロトコル"""

    def integrate(
        self,
        dt: float,
        derivative_fn: Callable[[Any, float], Any],
        state: Any,
        **kwargs,
    ) -> Any:
        """時間積分を実行

        Args:
            dt: 時間刻み幅
            derivative_fn: 時間微分を計算する関数
            state: 現在の状態
            **kwargs: 追加のパラメータ

        Returns:
            更新された状態
        """
        ...

    def get_diagnostics(self) -> Dict[str, Any]:
        """診断情報を取得"""
        ...
