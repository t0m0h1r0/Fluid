"""シミュレーションの実行を管理するモジュール

このモジュールは、シミュレーションの時間発展計算を管理します。
"""

import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from physics.navier_stokes.solver import NavierStokesSolver
from physics.levelset import LevelSetSolver
from physics.poisson.sor import SORSolver
from logger import SimulationLogger
from .state import SimulationState
from .monitor import SimulationMonitor

class SimulationRunner:
    """シミュレーションの実行を管理するクラス
    
    時間発展計算の実行、状態の保存、モニタリングを行います。
    """
    
    def __init__(self, config: Dict[str, Any],
                 logger: SimulationLogger,
                 monitor: Optional[SimulationMonitor] = None):
        """シミュレーションランナーを初期化
        
        Args:
            config: シミュレーション設定
            logger: ロガー
            monitor: シミュレーションモニター
        """
        self.config = config
        self.logger = logger.start_section("runner")
        self.monitor = monitor
        
        # Poissonソルバーの初期化
        poisson_config = config['numerical']['pressure_solver']
        poisson_solver = SORSolver(
            omega=poisson_config.get('omega', 1.5),
            tolerance=poisson_config.get('tolerance', 1e-6),
            max_iterations=poisson_config.get('max_iterations', 100)
        )
        
        # ソルバーの初期化
        self.ns_solver = NavierStokesSolver(
            poisson_solver=poisson_solver,
            use_weno=config.get('numerical', {}).get('use_weno', True)
        )
        self.ls_solver = LevelSetSolver()
        
        # 数値パラメータの設定
        self.max_time = config['numerical']['max_time']
        self.save_interval = config['numerical']['save_interval']
        
        # 停止条件のチェック用コールバック
        self._stop_check: Optional[Callable[[SimulationState], bool]] = None
    
    def run(self, state: SimulationState, 
            output_dir: Path,
            stop_check: Optional[Callable[[SimulationState], bool]] = None) -> SimulationState:
        """シミュレーションを実行
        
        Args:
            state: 初期状態
            output_dir: 出力ディレクトリ
            stop_check: 停止条件をチェックするコールバック関数
            
        Returns:
            最終状態
        """
        self.logger.info("シミュレーション開始")
        start_time = time.time()
        self._stop_check = stop_check
        
        try:
            while self._should_continue(state):
                state = self._advance_step(state)
                
                # 結果の保存
                if state.time >= state.next_save:
                    self._save_results(state, output_dir)
                    state.next_save += self.save_interval
                
                # モニタリング
                if self.monitor:
                    self.monitor.update(state)
            
            # 最終状態の保存
            self._save_results(state, output_dir)
            
            elapsed = time.time() - start_time
            self.logger.info(f"シミュレーション完了: {elapsed:.1f}秒")
            return state
            
        except Exception as e:
            self.logger.log_error_with_context(
                "シミュレーション実行中にエラーが発生",
                e,
                {'time': state.time, 'iteration': state.iteration}
            )
            raise
    
    def _should_continue(self, state: SimulationState) -> bool:
        """シミュレーションを継続するか判定
        
        Args:
            state: 現在の状態
            
        Returns:
            継続するかどうか
        """
        if not state.is_valid():
            self.logger.error("Invalid simulation state detected")
            return False
        
        if state.time >= self.max_time:
            self.logger.info("Maximum simulation time reached")
            return False
        
        if self._stop_check and self._stop_check(state):
            self.logger.info("Stop condition met")
            return False
        
        return True
    
    def _advance_step(self, state: SimulationState) -> SimulationState:
        """1ステップ進める
        
        Args:
            state: 現在の状態
            
        Returns:
            更新された状態
        """
        # 時間刻み幅の計算
        dt_ns = self.ns_solver.compute_timestep(
            state.velocity,
            properties=state.properties
        )
        dt_ls = self.ls_solver.compute_timestep(
            state.levelset,
            state.velocity
        )
        dt = min(dt_ns, dt_ls)
        
        # Navier-Stokes方程式の時間発展
        ns_result = self.ns_solver.advance(
            dt, state.velocity, state.pressure,
            levelset=state.levelset,
            properties=state.properties
        )
        
        # Level Set関数の移流
        ls_result = self.ls_solver.advance(
            dt, state.levelset, state.velocity
        )
        
        # 状態の更新
        state.velocity = ns_result['velocity']
        state.pressure = ns_result['pressure']
        state.time += dt
        state.iteration += 1
        
        # 統計情報の更新
        state.update_statistics()
        
        if state.iteration % 100 == 0:  # 定期的な進捗報告
            self.logger.info(
                f"Time: {state.time:.3f}, "
                f"Max div: {state.statistics['max_divergence']:.2e}, "
                f"Max vel: {state.statistics['max_velocity']:.2e}"
            )
        
        return state
    
    def _save_results(self, state: SimulationState, output_dir: Path):
        """結果を保存
        
        Args:
            state: 保存する状態
            output_dir: 出力ディレクトリ
        """
        save_dir = output_dir / f"time_{state.time:.6f}"
        state.save(save_dir)
        
        self.logger.info(f"Results saved at t = {state.time:.3f}")
    
    def checkpoint(self, state: SimulationState, path: Path):
        """チェックポイントを保存
        
        Args:
            state: 保存する状態
            path: 保存先パス
        """
        state.save(path / "checkpoint")
        self.logger.info(f"Checkpoint saved at iteration {state.iteration}")
    
    @classmethod
    def from_checkpoint(cls, path: Path, config: Dict[str, Any],
                       logger: SimulationLogger) -> tuple['SimulationRunner', SimulationState]:
        """チェックポイントから復元
        
        Args:
            path: チェックポイントのパス
            config: シミュレーション設定
            logger: ロガー
            
        Returns:
            (ランナー, 状態)のタプル
        """
        runner = cls(config, logger)
        state = SimulationState.load(path / "checkpoint", config)
        return runner, state