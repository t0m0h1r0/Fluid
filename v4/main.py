import os
import numpy as np
from config import SimulationConfig
from phase_field import PhaseField
from compact_solver import CompactSolver
from flow_solver import FlowSolver
from io_handler import SimulationIO
from visualizer import Visualizer
import matplotlib.pyplot as plt


def simulate(config, start_time=0.0, start_step=0, max_time=1.0):
    """シミュレーションの実行"""
    # 各コンポーネントの初期化
    phase = PhaseField(config)
    compact = CompactSolver(config)
    flow = FlowSolver(config, phase, compact)
    io = SimulationIO(config)
    vis = Visualizer(config)
    
    time = start_time
    step = start_step
    next_save = time + config.save_interval
    
    print(f"Starting simulation from t = {time}")
    
    while time < max_time:
        # 時間発展
        flow.runge_kutta_step()
        time += config.dt
        step += 1
        
        # 進捗表示
        if step % 100 == 0:
            print(f"t = {time:.3f}, step = {step}")
        
        # 定期保存と可視化
        if time >= next_save:
            print(f"Saving state at t = {time:.3f}")
            
            # 状態の保存
            io.save_state(step, time, flow, phase)
            io.save_for_visualization(step, time, flow, phase)
            
            # 可視化
            fig_phase = vis.plot_phase_field(phase.phi, time)
            fig_phase.savefig(f'output/phase_{step:06d}.png')
            plt.close(fig_phase)
            
            fig_vel = vis.plot_velocity(flow.u, flow.v, flow.w, time)
            fig_vel.savefig(f'output/velocity_{step:06d}.png')
            plt.close(fig_vel)
            
            fig_vec = vis.plot_velocity_vectors(flow.u, flow.v, flow.w, time)
            fig_vec.savefig(f'output/vectors_{step:06d}.png')
            plt.close(fig_vec)
            
            next_save = time + config.save_interval

def resume_simulation(config, checkpoint_file):
    """チェックポイントからシミュレーションを再開"""
    phase = PhaseField(config)
    compact = CompactSolver(config)
    flow = FlowSolver(config, phase, compact)
    io = SimulationIO(config)
    
    print(f"Resuming from checkpoint: {checkpoint_file}")
    time, step = io.load_state(checkpoint_file, flow, phase)
    
    simulate(config, time, step)

def main():
    # 出力ディレクトリの作成
    os.makedirs('output', exist_ok=True)
    
    # シミュレーション設定
    config = SimulationConfig()
    
    # コマンドライン引数のチェック
    import sys
    if len(sys.argv) > 1:
        # チェックポイントからの再開
        checkpoint_file = sys.argv[1]
        resume_simulation(config, checkpoint_file)
    else:
        # 新規シミュレーション開始
        simulate(config)

if __name__ == "__main__":
    main()
