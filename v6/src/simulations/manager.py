# simulation/manager.py
import numpy as np
from typing import Dict, Any, List, Optional
from core.interfaces import Field, PhysicalModel, NumericalScheme, BoundaryCondition, Solver
from solvers.poisson import AbstractPoissonSolver
from solvers.time_integration import TimeIntegrationSolver

class SimulationManager:
    """
    マルチフィールドシミュレーションを管理するクラス
    
    複数の物理場の相互作用と時間発展を統括
    """
    def __init__(self, 
                 fields: Dict[str, Field],
                 poisson_solver: Optional[AbstractPoissonSolver] = None,
                 time_integrator: Optional[TimeIntegrationSolver] = None):
        """
        シミュレーションマネージャーの初期化
        
        Args:
            fields: シミュレーションで扱う物理場（フィールド）の辞書
            poisson_solver: Poisson方程式ソルバー（オプション）
            time_integrator: 時間発展ソルバー（オプション）
        """
        self.fields = fields
        self.poisson_solver = poisson_solver
        self.time_integrator = time_integrator
        
        # シミュレーション状態
        self.current_time = 0.0
        self.timestep = 0
        
        # シミュレーション設定
        self.max_timesteps = 1000
        self.max_time = 10.0
        self.dt = 0.01
    
    def prepare_context(self) -> Dict[str, Any]:
        """
        時間発展に必要なコンテキスト情報を準備
        
        Returns:
            シミュレーションコンテキスト
        """
        context = {}
        for name, field in self.fields.items():
            context[name] = field.state
        return context
    
    def advance_timestep(self) -> Dict[str, Any]:
        """
        1タイムステップ進める
        
        Returns:
            各フィールドの更新情報
        """
        # コンテキストの準備
        context = self.prepare_context()
        
        # 各フィールドの時間発展
        updated_fields = {}
        for name, field in self.fields.items():
            # 時間発展
            updated_fields[name] = field.advance(self.dt, context)
        
        # シミュレーション状態の更新
        self.current_time += self.dt
        self.timestep += 1
        
        return updated_fields
    
    def run_simulation(self, 
                      output_interval: Optional[int] = None,
                      callback: Optional[callable] = None):
        """
        シミュレーションの実行
        
        Args:
            output_interval: 出力間隔（タイムステップ）
            callback: 各タイムステップ後に呼び出される関数
        """
        print("シミュレーション開始")
        
        while (self.timestep < self.max_timesteps and 
               self.current_time < self.max_time):
            # タイムステップの進行
            updated_fields = self.advance_timestep()
            
            # 出力とコールバック
            if (output_interval and 
                self.timestep % output_interval == 0):
                self._output_results(updated_fields)
            
            # ユーザー定義のコールバック
            if callback:
                callback(self.timestep, self.current_time, updated_fields)
            
            # 収束判定や終了条件の確認
            if self._check_termination(updated_fields):
                break
        
        print("シミュレーション終了")
    
    def _output_results(self, updated_fields: Dict[str, Any]):
        """
        結果の出力
        
        Args:
            updated_fields: 更新されたフィールド
        """
        print(f"タイムステップ: {self.timestep}, 時間: {self.current_time:.4f}")
        for name, field in updated_fields.items():
            print(f"{name}フィールドの統計:")
            print(f"  最小値: {np.min(field):.4f}")
            print(f"  最大値: {np.max(field):.4f}")
            print(f"  平均値: {np.mean(field):.4f}")
    
    def _check_termination(self, updated_fields: Dict[str, Any]) -> bool:
        """
        シミュレーション終了条件の確認
        
        Args:
            updated_fields: 更新されたフィールド
        
        Returns:
            終了すべきかどうか
        """
        # エネルギー保存則のチェックなど
        # 具体的な終了条件は物理モデルに依存
        return False
    
    def get_field(self, name: str) -> Field:
        """
        指定された名前のフィールドを取得
        
        Args:
            name: フィールド名
        
        Returns:
            対応するフィールド
        """
        return self.fields[name]
