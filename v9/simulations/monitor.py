"""シミュレーションの進行状況を監視・記録するモジュール"""

import json
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt

from simulations.state import SimulationState


class SimulationMonitor:
    """シミュレーションの進行状況を監視・記録するクラス"""

    def __init__(self, config: Dict[str, Any], logger=None):
        """初期化

        Args:
            config: シミュレーション設定
            logger: ロガー
        """
        self.config = config
        self.logger = logger

        # 統計情報の初期化
        self.statistics = {
            "time_history": [],
            "velocity_magnitude": [],
            "pressure_max": [],
            "levelset_volume": [],
            "levelset_area": [],
        }

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント

        Returns:
            SimulationMonitorインスタンス
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理

        Args:
            exc_type: 例外の型
            exc_val: 例外のインスタンス
            exc_tb: トレースバック
        """
        # 例外の種類に関わらず、統計情報の保存などの終了処理を実行
        if self.logger:
            self.logger.info("シミュレーションモニターを終了")

        # 任意の終了処理を追加できます
        return False  # 例外を再送出

    def update(self, state: SimulationState):
        """状態を更新

        Args:
            state: 現在のシミュレーション状態
        """
        try:
            # 時間を取得（速度場から取得できない場合は、状態の時間を使用）
            current_time = state.time if hasattr(state, 'time') else 0.0

            # 速度の大きさを計算
            velocity_magnitude = state.velocity.magnitude().mean()

            # レベルセットの診断情報を取得
            levelset_diag = state.levelset.get_diagnostics()

            # 統計情報を記録
            self.statistics["time_history"].append(current_time)
            self.statistics["velocity_magnitude"].append(velocity_magnitude)
            self.statistics["pressure_max"].append(state.pressure.max())
            self.statistics["levelset_volume"].append(levelset_diag["volume"])
            self.statistics["levelset_area"].append(levelset_diag["area"])

        except Exception as e:
            if self.logger:
                self.logger.warning(f"モニター更新中にエラー: {e}")

    def plot_history(self, output_dir: Path):
        """シミュレーション履歴をプロット

        Args:
            output_dir: 出力ディレクトリ
        """
        try:
            # プロット用のディレクトリを作成
            plot_dir = output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            # 速度の大きさをプロット
            plt.figure(figsize=(10, 5))
            plt.plot(
                self.statistics["time_history"], 
                self.statistics["velocity_magnitude"]
            )
            plt.title("Velocity Magnitude")
            plt.xlabel("Time")
            plt.ylabel("Mean Velocity")
            plt.tight_layout()
            plt.savefig(plot_dir / "velocity_magnitude.png")
            plt.close()

            # 圧力の最大値をプロット
            plt.figure(figsize=(10, 5))
            plt.plot(
                self.statistics["time_history"], 
                self.statistics["pressure_max"]
            )
            plt.title("Maximum Pressure")
            plt.xlabel("Time")
            plt.ylabel("Max Pressure")
            plt.tight_layout()
            plt.savefig(plot_dir / "pressure_max.png")
            plt.close()

            # レベルセットの体積をプロット
            plt.figure(figsize=(10, 5))
            plt.plot(
                self.statistics["time_history"], 
                self.statistics["levelset_volume"]
            )
            plt.title("Level Set Volume")
            plt.xlabel("Time")
            plt.ylabel("Volume")
            plt.tight_layout()
            plt.savefig(plot_dir / "levelset_volume.png")
            plt.close()

        except Exception as e:
            if self.logger:
                self.logger.warning(f"履歴プロットの生成中にエラーが発生: {e}")

    def generate_report(self, output_dir: Path):
        """シミュレーション結果のレポートを生成

        Args:
            output_dir: 出力ディレクトリ
        """
        try:
            # 統計情報をJSONファイルに保存
            report_path = output_dir / "statistics.json"
            with open(report_path, "w") as f:
                json.dump(self.statistics, f, indent=2)

            if self.logger:
                self.logger.info(f"統計情報を保存: {report_path}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"レポート生成中にエラーが発生: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """シミュレーションの概要を取得

        Returns:
            シミュレーション概要
        """
        return {
            "total_time_steps": len(self.statistics["time_history"]),
            "final_time": self.statistics["time_history"][-1]
            if self.statistics["time_history"]
            else 0,
            "max_velocity": max(self.statistics["velocity_magnitude"])
            if self.statistics["velocity_magnitude"]
            else 0,
            "max_pressure": max(self.statistics["pressure_max"])
            if self.statistics["pressure_max"]
            else 0,
        }