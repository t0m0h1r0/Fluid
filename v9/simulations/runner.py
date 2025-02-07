"""シミュレーションの実行を管理するモジュール

Navier-Stokes方程式のソルバーを用いて、シミュレーションを実行します。
"""

import numpy as np
from typing import Dict, Any
from pathlib import Path

from logger import SimulationLogger
from simulations.monitor import SimulationMonitor
from physics.navier_stokes import NavierStokesSolver
from physics.poisson import SORSolver
from core.field import VectorField, ScalarField
from physics.levelset import LevelSetField


class SimulationRunner:
    """シミュレーションの実行を管理するクラス

    Navier-Stokes方程式を解き、シミュレーションを進行させます。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: SimulationLogger,
        monitor: SimulationMonitor,
    ):
        """シミュレーションランナーを初期化

        Args:
            config: シミュレーション設定
            logger: ロガー
            monitor: モニター
        """
        self.config = config
        self.logger = logger
        self.monitor = monitor

        # 数値スキーム設定の取得
        numerical_config = config.get("numerical", {})

        # 最大シミュレーション時間の取得
        self.max_time = numerical_config.get("max_time", 2.0)

        # 初期時間刻みの取得
        self.initial_dt = numerical_config.get("initial_dt", 0.001)

        # 保存間隔の取得
        self.save_interval = numerical_config.get("save_interval", 0.1)

        # 圧力ソルバーの設定
        pressure_solver_config = numerical_config.get("pressure_solver", {})

        # SORソルバーの初期化
        poisson_solver = SORSolver(
            omega=pressure_solver_config.get("omega", 1.5),
            max_iterations=pressure_solver_config.get("max_iterations", 200),
            tolerance=pressure_solver_config.get("tolerance", 1.0e-8),
        )

        # Navier-Stokesソルバーの初期化
        self.ns_solver = NavierStokesSolver(
            logger=logger, poisson_solver=poisson_solver
        )

        # デバッグ設定の取得
        self.debug_config = config.get("debug", {})

    def run(self, initial_state, output_dir: Path):
        """シミュレーションを実行

        Args:
            initial_state: 初期状態
            output_dir: 出力ディレクトリ

        Returns:
            最終状態
        """
        self.logger.info("シミュレーション開始")
        current_state = initial_state
        current_time = 0.0
        iteration = 0

        # 出力ディレクトリの作成
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 結果の保存先
        results_dir = output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        try:
            # メインシミュレーションループ
            while current_time < self.max_time:
                # 時間刻みの計算
                dt = self._compute_timestep(current_state)

                # 1ステップ進める
                current_state = self._advance_step(current_state)

                # 時間と反復回数を更新
                current_time += dt
                iteration += 1

                # 結果の保存
                if iteration % int(self.save_interval / dt) == 0:
                    self._save_state(current_state, results_dir, current_time)

                # モニターへの更新
                self.monitor.update(current_state, current_time)

                # デバッグ出力
                if self.debug_config.get("check_divergence", False):
                    self._check_divergence(current_state)

                # プロファイリング
                if self.debug_config.get("profiling", False):
                    self._profile_simulation(current_state)

        except Exception as e:
            self.logger.error(f"シミュレーション実行中にエラーが発生: {e}")
            raise

        self.logger.info("シミュレーション終了")
        return current_state

    def _compute_timestep(self, state):
        """適応的な時間刻みを計算

        Args:
            state: 現在の状態

        Returns:
            時間刻み
        """
        # Navier-Stokesソルバーから推奨される時間刻みを取得
        recommended_dt = self.ns_solver.compute_timestep(state)

        # CFL条件に基づいて調整
        numerical_config = self.config.get("numerical", {})
        cfl = numerical_config.get("time_integration", {}).get("cfl", 0.3)

        return min(self.initial_dt, cfl * recommended_dt)

    def _advance_step(self, state):
        """1時間ステップを進める

        Args:
            state: 現在の状態

        Returns:
            更新された状態
        """
        # Navier-Stokesソルバーで1ステップ進める
        ns_result = self.ns_solver.advance(
            state,
            max_iterations=200,  # 必要に応じて調整
        )

        return ns_result

    def _save_state(self, state, results_dir, current_time):
        """状態を保存

        Args:
            state: 現在の状態
            results_dir: 結果の保存先ディレクトリ
            current_time: 現在の時刻
        """
        # VTK形式で保存
        try:
            # 必要に応じてPyEVTKをインポート
            from pyevtk.hl import gridToVTK

            # 保存するフィールド
            fields_to_save = {
                "velocity_x": state.velocity.components[0].data,
                "velocity_y": state.velocity.components[1].data,
                "pressure": state.pressure.data,
                "levelset": state.levelset.data,
            }

            # ファイルパスの生成
            filepath = results_dir / f"state_{current_time:.6f}"

            # グリッド座標の生成
            shape = list(list(fields_to_save.values())[0].shape)
            x = np.linspace(0, 1, shape[0])
            y = np.linspace(0, 1, shape[1])
            z = np.linspace(0, 1, shape[2])

            # VTKファイルに出力
            gridToVTK(str(filepath), x, y, z, pointData=fields_to_save)

        except ImportError:
            self.logger.warning(
                "PyEVTKがインストールされていないため、状態の保存をスキップします。"
            )
        except Exception as e:
            self.logger.error(f"状態の保存中にエラーが発生: {e}")

    def _check_divergence(self, state):
        """発散をチェック

        Args:
            state: 現在の状態
        """
        # 速度場の発散を計算
        divergence = state.velocity.divergence()
        max_divergence = np.max(np.abs(divergence.data))

        if max_divergence > 1e-3:
            self.logger.warning(f"大きな速度発散を検出: {max_divergence}")

    def _profile_simulation(self, state):
        """シミュレーションのプロファイリング

        Args:
            state: 現在の状態
        """
        # プロファイリング情報の収集
        profile_info = {
            "velocity_magnitude": np.max(
                np.sqrt(sum(c.data**2 for c in state.velocity.components))
            ),
            "pressure_range": (
                np.min(state.pressure.data),
                np.max(state.pressure.data),
            ),
            "levelset_interface_width": np.sum(state.levelset.delta()),
        }

        self.logger.info(f"プロファイリング情報: {profile_info}")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, config, logger):
        """チェックポイントからシミュレーションを再開

        Args:
            checkpoint_path: チェックポイントファイルのパス
            config: シミュレーション設定
            logger: ロガー

        Returns:
            (SimulationRunner, 復元された状態)のタプル
        """
        # モニターの作成
        monitor = SimulationMonitor(config, logger)

        # ランナーの作成
        runner = cls(config, logger, monitor)

        # チェックポイントファイルの読み込み
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = np.load(f, allow_pickle=True)

        # 状態の復元
        velocity = VectorField(checkpoint_data["velocity_shape"])
        velocity.components[0].data = checkpoint_data["velocity_x"]
        velocity.components[1].data = checkpoint_data["velocity_y"]

        pressure = ScalarField(checkpoint_data["pressure_shape"])
        pressure.data = checkpoint_data["pressure"]

        levelset = LevelSetField(checkpoint_data["levelset_shape"])
        levelset.data = checkpoint_data["levelset"]

        # 状態のフィールドを持つクラスを作成（必要に応じて調整）
        class SimulationState:
            def __init__(self, velocity, pressure, levelset):
                self.velocity = velocity
                self.pressure = pressure
                self.levelset = levelset

        restored_state = SimulationState(velocity, pressure, levelset)

        return runner, restored_state
