from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

from core.field import VectorField
from physics.levelset import LevelSetField
from physics.properties import PropertiesManager, FluidProperties

from .ns_evolution import NavierStokesEvolution
from .ls_evolution import LevelSetEvolution
from visualization import visualize_simulation_state


class TwoPhaseSimulation:
    """二相流シミュレーションを管理するクラス"""

    def __init__(
        self,
        dimensions: tuple = (64, 64, 64),
        domain_size: tuple = (1.0, 1.0, 1.0),
        initial_conditions: Dict[str, Any] = None,
        max_time: float = 1.0,
        output_dir: Optional[str] = "results",
        save_interval: float = 0.1,
    ):
        """シミュレーションを初期化

        Args:
            dimensions: グリッドの次元
            domain_size: 領域のサイズ
            initial_conditions: 初期条件の設定
            max_time: シミュレーションの最大時間
            output_dir: 出力ディレクトリ
            save_interval: 結果保存の間隔
        """
        # ディレクトリの設定
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # デフォルトの初期条件
        default_conditions = {
            "height_fraction": 0.8,
            "water_phase": {
                "density": 1000.0,
                "viscosity": 1.0e-3,
                "surface_tension": 0.07,
            },
            "nitrogen_phase": {
                "density": 1.25,
                "viscosity": 1.81e-5,
                "surface_tension": 0.0,
            },
            "objects": [
                {
                    "type": "sphere",
                    "phase": "nitrogen",
                    "center": [0.5, 0.5, 0.4],
                    "radius": 0.2,
                }
            ],
        }
        self.initial_conditions = initial_conditions or default_conditions

        # シミュレーションパラメータ
        self.dimensions = dimensions
        self.domain_size = domain_size
        self.max_time = max_time
        self.save_interval = save_interval

        # 物性値マネージャーの設定
        water_props = FluidProperties(
            density=self.initial_conditions["water_phase"]["density"],
            viscosity=self.initial_conditions["water_phase"]["viscosity"],
            surface_tension=self.initial_conditions["water_phase"]["surface_tension"],
        )
        nitrogen_props = FluidProperties(
            density=self.initial_conditions["nitrogen_phase"]["density"],
            viscosity=self.initial_conditions["nitrogen_phase"]["viscosity"],
            surface_tension=self.initial_conditions["nitrogen_phase"][
                "surface_tension"
            ],
        )
        self.properties_manager = PropertiesManager(
            phase1=water_props, phase2=nitrogen_props
        )

        # 速度場とレベルセット場の初期化
        self._initialize_fields()

        # 進化クラスの初期化
        self.ns_evolution = NavierStokesEvolution(
            properties_manager=self.properties_manager
        )
        self.ls_evolution = LevelSetEvolution()

        # シミュレーション状態
        self.current_time = 0.0
        self.current_dt = 0.0
        self.iteration = 0

    def _initialize_fields(self):
        """初期フィールドを設定"""
        # グリッド間隔の計算
        dx = self.domain_size[0] / self.dimensions[0]

        # 速度場の初期化（ゼロ）
        self.velocity = VectorField(self.dimensions, dx)

        # レベルセット場の初期化
        self.level_set = LevelSetField(self.dimensions, dx)

        # 水面の高さ
        water_height = self.initial_conditions["height_fraction"] * self.domain_size[2]

        # 各オブジェクトの初期化
        for obj in self.initial_conditions.get("objects", []):
            if obj["type"] == "sphere":
                self._initialize_sphere(obj, water_height)

    def _initialize_sphere(self, obj: Dict[str, Any], water_height: float):
        """球体の初期化

        Args:
            obj: 球体の設定
            water_height: 水面の高さ
        """
        # 球体の中心と半径の計算
        center = [c * d for c, d in zip(obj["center"], self.domain_size)]
        radius = obj["radius"] * self.domain_size[0]  # x方向のサイズを使用

        # メッシュグリッドの生成
        x = np.linspace(0, self.domain_size[0], self.dimensions[0])
        y = np.linspace(0, self.domain_size[1], self.dimensions[1])
        z = np.linspace(0, self.domain_size[2], self.dimensions[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # レベルセット関数の計算
        signed_distance = radius - np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )

        # 水面の影響を考慮
        water_zone = Z <= water_height
        signed_distance[~water_zone] = -np.abs(signed_distance[~water_zone])

        # フェーズの設定（窒素球か水球か）
        sign = 1 if obj["phase"] == "water" else -1
        self.level_set.data = sign * signed_distance

    def run(self):
        """シミュレーションを実行"""
        # 初期状態の可視化
        self._save_state(0)

        # メインシミュレーションループ
        while self.current_time < self.max_time:
            # 時間ステップの計算
            self.current_dt = min(
                self.ns_evolution.compute_timestep(self.velocity, self.level_set),
                self.ls_evolution.compute_timestep(self.velocity, self.level_set),
            )

            # Level Set場の時間発展
            ls_result = self.ls_evolution.advance(
                self.velocity, self.level_set, self.current_dt
            )

            # Navier-Stokes場の時間発展
            ns_result = self.ns_evolution.advance(
                self.velocity, self.level_set, self.current_dt
            )

            self.level_set = ls_result["level_set"]
            self.velocity = ns_result["velocity"]

            # 時間と反復回数の更新
            self.current_time += self.current_dt
            self.iteration += 1

            # 状態の保存
            if self.iteration % int(self.save_interval / self.current_dt) == 0:
                self._save_state(self.current_time)

        # 最終状態の保存
        self._save_state(self.current_time)

    def _save_state(self, timestamp: float):
        """シミュレーション状態を保存

        Args:
            timestamp: 現在の時刻
        """
        # 可視化関数を使用して状態を保存
        visualize_simulation_state(
            state=self,  # StateインターフェースでサポートされるようSimulationクラスを調整
            config=self._get_visualization_config(),
            timestamp=timestamp,
        )

    def _get_visualization_config(self) -> Dict[str, Any]:
        """可視化設定を取得

        Returns:
            可視化設定の辞書
        """
        return {
            "output_dir": str(self.output_dir),
            "format": "png",
            "dpi": 300,
            "fields": {
                "velocity": {"enabled": True},
                "pressure": {"enabled": True},
                "levelset": {"enabled": True},
            },
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """シミュレーションの診断情報を取得

        Returns:
            診断情報の辞書
        """
        return {
            "time": self.current_time,
            "iteration": self.iteration,
            "time_step": self.current_dt,
            "velocity_stats": {
                "max": max(np.max(np.abs(v.data)) for v in self.velocity.components),
                "min": min(np.min(np.abs(v.data)) for v in self.velocity.components),
            },
        }


# メイン実行用のエントリーポイント
def main():
    """シミュレーションのメイン関数"""
    # デフォルトのシミュレーションを実行
    sim = TwoPhaseSimulation()
    sim.run()


if __name__ == "__main__":
    main()
