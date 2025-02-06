from typing import Dict, Any, Optional
from pathlib import Path

from utils.config import SimulationConfig
from data_io.data_manager import DataManager
from visualization.visualizer import Visualizer

from .state import SimulationState
from .initialization import SimulationInitializer
from .time_evolution import TimeEvolutionManager
from .statistics import StatisticsAnalyzer
from .diagnostics import DiagnosticsAnalyzer

class SimulationRunner:
    """シミュレーション全体を制御するクラス"""
    
    def __init__(self, config_file: str):
        """
        Args:
            config_file: 設定ファイルのパス
        """
        # 設定の読み込み
        self.config = SimulationConfig(config_file)
        
        # シミュレーション状態の初期化
        self.state = SimulationState()
        
        # 各コンポーネントの初期化
        self._initialize_components()
        
        # フィールドとソルバーの初期化
        self._initialize_simulation()
    
    def _initialize_components(self):
        """コンポーネントの初期化"""
        # 初期化マネージャー
        self.initializer = SimulationInitializer(self.config)
        
        # データ管理とビジュアライザ
        self.data_manager = DataManager(self.config.output_dir)
        self.visualizer = Visualizer(self.config.output_dir)
        
        # 各種マネージャー
        self.evolution_manager = TimeEvolutionManager(self.config)
        self.stats_analyzer = StatisticsAnalyzer(self.config)
        self.diagnostics_analyzer = DiagnosticsAnalyzer(self.config)
    
    def _initialize_simulation(self):
        """シミュレーションの初期化"""
        try:
            # フィールドとソルバーの初期化
            self.fields, self.solvers = self.initializer.initialize()
            
            # 統計情報の初期化
            self.stats_analyzer.initialize(self.fields)
            
            # 初期状態の診断
            initial_diagnostics = self.diagnostics_analyzer.run_diagnostics(
                self.fields,
                self.stats_analyzer.get_current_stats()
            )
            
            # 初期状態の保存と可視化
            self._save_current_state()
            
            # 初期化の記録
            self.data_manager.save_log("シミュレーションを初期化しました")
            self.data_manager.save_log(
                self.diagnostics_analyzer.get_summary(initial_diagnostics)
            )
            
        except Exception as e:
            self.data_manager.save_log(
                f"初期化中にエラーが発生: {str(e)}",
                level='ERROR'
            )
            raise
    
    def run(self):
        """シミュレーションの実行"""
        try:
            self.state.is_running = True
            self.data_manager.save_log("シミュレーションを開始します")
            
            while self._should_continue():
                self._advance_timestep()
                
                # 定期的な保存と診断
                if self._should_save():
                    self._save_and_diagnose()
                
                # 収束判定
                if self._check_convergence():
                    self.data_manager.save_log("収束条件を満たしました")
                    break
            
            # 最終状態の保存
            self._save_current_state()
            self.data_manager.save_log("シミュレーションが完了しました")
            
        except KeyboardInterrupt:
            self.data_manager.save_log("シミュレーションが中断されました")
            self._save_current_state()
            
        except Exception as e:
            self.data_manager.save_log(
                f"シミュレーション中にエラーが発生: {str(e)}",
                level='ERROR'
            )
            raise
            
        finally:
            self.state.finish()
    
    def _advance_timestep(self):
        """1ステップの時間発展"""
        # 時間発展の実行
        result = self.evolution_manager.advance_timestep(
            self.fields,
            self.solvers
        )
        
        # 結果の更新
        self.fields = result.fields
        self.state.update_time(result.dt)
        
        # 統計情報の更新
        self.stats_analyzer.update(self.fields, result.dt)
    
    def _should_continue(self) -> bool:
        """シミュレーション継続判定"""
        return (
            self.state.is_running and
            self.state.current_time < self.config.time.max_time and
            self.state.current_step < self.config.time.max_steps
        )
    
    def _should_save(self) -> bool:
        """保存判定"""
        return self.state.should_save(self.config.time.save_interval)
    
    def _check_convergence(self) -> bool:
        """収束判定"""
        if not self.state.current_step > 0:
            return False
            
        return self.stats_analyzer.check_convergence(self.config.convergence_criteria)
    
    def _save_current_state(self):
        """現在の状態を保存"""
        # 統計情報の取得
        stats = self.stats_analyzer.get_current_stats()
        
        # データの保存
        self.data_manager.save_state(
            self.state.current_step,
            self.state.current_time,
            self.fields,
            stats=stats
        )
        
        # 可視化の作成
        self.visualizer.create_visualization(
            self.fields,
            self.state.current_time,
            self.state.current_step,
            stats=stats
        )
        
        self.state.mark_save()
    
    def _save_and_diagnose(self):
        """状態の保存と診断の実行"""
        # 現在の状態を保存
        self._save_current_state()
        
        # 診断の実行
        diagnostics = self.diagnostics_analyzer.run_diagnostics(
            self.fields,
            self.stats_analyzer.get_current_stats()
        )
        
        # 診断結果の記録
        self.data_manager.save_log(
            f"\nステップ {self.state.current_step}: "
            f"t = {self.state.current_time:.3f}s\n" +
            self.diagnostics_analyzer.get_summary(diagnostics)
        )
    
    def get_status(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return {
            'simulation_state': self.state.get_status(),
            'statistics': self.stats_analyzer.get_current_stats(),
            'elapsed_time': self.state.get_elapsed_time()
        }
    
    def pause(self):
        """シミュレーションの一時停止"""
        self.state.is_running = False
        self.data_manager.save_log("シミュレーションを一時停止しました")
    
    def resume(self):
        """シミュレーションの再開"""
        if not self.state.is_running:
            self.state.is_running = True
            self.data_manager.save_log("シミュレーションを再開しました")
            self.run()
    
    def abort(self):
        """シミュレーションの中断"""
        self.state.is_running = False
        self._save_current_state()
        self.data_manager.save_log("シミュレーションを中断しました")

def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='二相流体シミュレーション')
    parser.add_argument('--config', default='config/simulation.yaml',
                       help='設定ファイルのパス')
    args = parser.parse_args()
    
    # シミュレーションの実行
    runner = SimulationRunner(args.config)
    runner.run()

if __name__ == '__main__':
    main()
