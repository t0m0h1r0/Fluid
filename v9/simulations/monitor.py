"""シミュレーションの監視を行うモジュール

このモジュールは、シミュレーションの進行状況を監視し、
統計情報の収集や異常検知を行います。
"""

from typing import Dict, Any, List, Callable
from pathlib import Path
import time
import json
import numpy as np
from logger import SimulationLogger
from .state import SimulationState


class SimulationMonitor:
    """シミュレーション監視クラス

    シミュレーションの進行状況を監視し、統計情報の収集、
    異常検知、パフォーマンス測定などを行います。
    """

    def __init__(self, config: Dict[str, Any], logger: SimulationLogger):
        """モニターを初期化

        Args:
            config: モニター設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger.start_section("monitor")

        # 統計情報の保存先
        self.stats_file = (
            Path(config["visualization"]["output_dir"]) / "statistics.json"
        )

        # 履歴データ
        self.history: Dict[str, List[float]] = {
            "time": [],
            "wall_time": [],
            "max_velocity": [],
            "mean_velocity": [],
            "max_pressure": [],
            "kinetic_energy": [],
            "interface_area": [],
            "phase1_volume": [],
            "max_divergence": [],
        }

        # パフォーマンス測定
        self.start_time = time.time()
        self.last_update = self.start_time
        self.update_count = 0

        # 異常検知のしきい値
        self.divergence_threshold = config.get("debug", {}).get(
            "divergence_threshold", 1e-3
        )
        self.energy_growth_threshold = config.get("debug", {}).get(
            "energy_growth_threshold", 1.5
        )

        # カスタムコールバック
        self.custom_callbacks: List[Callable[[SimulationState], None]] = []

    def update(self, state: SimulationState):
        """状態を更新

        Args:
            state: 現在のシミュレーション状態
        """
        current_time = time.time()

        # 統計情報の記録
        self._record_statistics(state, current_time)

        # 異常検知
        self._check_anomalies(state)

        # カスタムコールバックの実行
        for callback in self.custom_callbacks:
            callback(state)

        # パフォーマンス統計の更新
        self._update_performance_stats(current_time)

        # 定期的な保存
        if self.update_count % 10 == 0:
            self.save_statistics()

    def add_callback(self, callback: Callable[[SimulationState], None]):
        """カスタムコールバックを追加

        Args:
            callback: 追加するコールバック関数
        """
        self.custom_callbacks.append(callback)

    def _record_statistics(self, state: SimulationState, current_time: float):
        """統計情報を記録

        Args:
            state: シミュレーション状態
            current_time: 現在の壁時計時間
        """
        stats = state.statistics

        self.history["time"].append(state.time)
        self.history["wall_time"].append(current_time - self.start_time)

        for key in stats:
            if key in self.history:
                self.history[key].append(stats[key])

    def _check_anomalies(self, state: SimulationState):
        """異常を検知

        Args:
            state: シミュレーション状態
        """
        # 発散のチェック
        if state.statistics["max_divergence"] > self.divergence_threshold:
            self.logger.warning(
                f"High divergence detected: {state.statistics['max_divergence']:.2e}"
            )

        # エネルギー増加のチェック
        if len(self.history["kinetic_energy"]) > 1:
            energy_ratio = (
                state.statistics["kinetic_energy"] / self.history["kinetic_energy"][0]
            )
            if energy_ratio > self.energy_growth_threshold:
                self.logger.warning(
                    f"Significant energy growth detected: {energy_ratio:.2f}x"
                )

    def _update_performance_stats(self, current_time: float):
        """パフォーマンス統計を更新

        Args:
            current_time: 現在の壁時計時間
        """
        dt = current_time - self.last_update
        steps_per_second = 1.0 / dt if dt > 0 else 0.0

        if self.update_count % 10 == 0:
            self.logger.info(f"Performance: {steps_per_second:.1f} steps/s")

        self.last_update = current_time
        self.update_count += 1

    def save_statistics(self):
        """統計情報をファイルに保存"""
        try:
            with self.stats_file.open("w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save statistics: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """シミュレーションのサマリーを取得

        Returns:
            サマリー情報を含む辞書
        """
        elapsed = time.time() - self.start_time

        return {
            "total_time": self.history["time"][-1],
            "wall_time": elapsed,
            "steps": self.update_count,
            "steps_per_second": self.update_count / elapsed,
            "max_divergence": max(self.history["max_divergence"]),
            "final_energy": self.history["kinetic_energy"][-1],
            "energy_ratio": (
                self.history["kinetic_energy"][-1] / self.history["kinetic_energy"][0]
            ),
        }

    def plot_history(self, output_dir: Path):
        """履歴データをプロット

        Args:
            output_dir: 出力ディレクトリ
        """
        try:
            import matplotlib.pyplot as plt

            # 時間発展のプロット
            plt.figure(figsize=(10, 6))
            plt.plot(
                self.history["time"],
                self.history["kinetic_energy"],
                label="Kinetic Energy",
            )
            plt.xlabel("Time")
            plt.ylabel("Energy")
            plt.yscale("log")
            plt.grid(True)
            plt.legend()
            plt.savefig(output_dir / "energy_history.png")
            plt.close()

            # 発散のプロット
            plt.figure(figsize=(10, 6))
            plt.semilogy(
                self.history["time"],
                self.history["max_divergence"],
                label="Max Divergence",
            )
            plt.xlabel("Time")
            plt.ylabel("Divergence")
            plt.grid(True)
            plt.legend()
            plt.savefig(output_dir / "divergence_history.png")
            plt.close()

            # 体積保存のプロット
            plt.figure(figsize=(10, 6))
            volume_ratio = (
                np.array(self.history["phase1_volume"])
                / self.history["phase1_volume"][0]
            )
            plt.plot(self.history["time"], volume_ratio, label="Volume Ratio")
            plt.xlabel("Time")
            plt.ylabel("Volume Ratio")
            plt.grid(True)
            plt.legend()
            plt.savefig(output_dir / "volume_conservation.png")
            plt.close()

        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            self.logger.warning(f"Failed to create plots: {e}")

    def generate_report(self, output_dir: Path):
        """シミュレーションレポートを生成

        Args:
            output_dir: 出力ディレクトリ
        """
        report_file = output_dir / "simulation_report.md"
        summary = self.get_summary()

        try:
            with report_file.open("w") as f:
                f.write("# Simulation Report\n\n")

                f.write("## Summary\n\n")
                f.write(f"- Total simulation time: {summary['total_time']:.3f} s\n")
                f.write(f"- Wall clock time: {summary['wall_time']:.1f} s\n")
                f.write(f"- Total steps: {summary['steps']}\n")
                f.write(f"- Performance: {summary['steps_per_second']:.1f} steps/s\n\n")

                f.write("## Conservation\n\n")
                f.write(f"- Energy ratio: {summary['energy_ratio']:.3f}\n")
                f.write(f"- Maximum divergence: {summary['max_divergence']:.2e}\n\n")

                f.write("## Visualization\n\n")
                f.write("![Energy History](energy_history.png)\n\n")
                f.write("![Divergence History](divergence_history.png)\n\n")
                f.write("![Volume Conservation](volume_conservation.png)\n")

            self.logger.info(f"Report generated: {report_file}")

        except Exception as e:
            self.logger.warning(f"Failed to generate report: {e}")

    def __enter__(self):
        """コンテキストマネージャーのエントリー"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーのイグジット"""
        self.save_statistics()
        if exc_type is not None:
            self.logger.error(f"Simulation terminated with error: {exc_val}")
        return False
