"""診断情報の収集と管理を提供するモジュール"""

from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class SimulationDiagnostics:
    """シミュレーションの診断情報を保持"""

    time: float = 0.0
    dt: float = 0.0
    navier_stokes: Dict[str, Any] = field(default_factory=dict)
    level_set: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)


class DiagnosticsCollector:
    """診断情報の収集と管理を担当"""

    def __init__(self, max_history: int = 1000):
        """診断情報コレクターを初期化

        Args:
            max_history: 保持する履歴の最大数
        """
        self.max_history = max_history
        self.diagnostics = SimulationDiagnostics()

    def collect(
        self,
        time: float,
        dt: float,
        navier_stokes: Dict[str, Any],
        level_set: Dict[str, Any],
    ) -> Dict[str, Any]:
        """診断情報を収集

        Args:
            time: 現在の時刻
            dt: 時間刻み幅
            navier_stokes: Navier-Stokesソルバーの診断情報
            level_set: Level Setソルバーの診断情報

        Returns:
            収集された診断情報
        """
        # 現在の診断情報を更新
        self.diagnostics.time = time
        self.diagnostics.dt = dt
        self.diagnostics.navier_stokes = navier_stokes
        self.diagnostics.level_set = level_set

        # 診断情報をまとめる
        current_diagnostics = {
            "time": time,
            "dt": dt,
            "navier_stokes": navier_stokes,
            "level_set": level_set,
        }

        # 履歴に追加
        self.diagnostics.history.append(current_diagnostics)
        if len(self.diagnostics.history) > self.max_history:
            self.diagnostics.history.pop(0)

        return current_diagnostics

    def get_current(self) -> Dict[str, Any]:
        """現在の診断情報を取得"""
        return {
            "time": self.diagnostics.time,
            "dt": self.diagnostics.dt,
            "navier_stokes": self.diagnostics.navier_stokes,
            "level_set": self.diagnostics.level_set,
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """診断情報の履歴を取得"""
        return self.diagnostics.history.copy()

    def get_time_series(self, key: str) -> List[float]:
        """特定の値の時系列を取得

        Args:
            key: 取得する値のキー（例: "navier_stokes.max_velocity"）

        Returns:
            時系列データ

        Raises:
            KeyError: 指定されたキーが見つからない場合
        """
        values = []
        keys = key.split(".")

        for diag in self.diagnostics.history:
            try:
                value = diag
                for k in keys:
                    value = value[k]
                values.append(float(value))
            except (KeyError, TypeError):
                raise KeyError(f"診断情報に{key}が見つかりません")

        return values

    def clear(self):
        """診断情報をクリア"""
        self.diagnostics = SimulationDiagnostics()
