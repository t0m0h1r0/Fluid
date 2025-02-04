import numpy as np
import h5py
from datetime import datetime

class SimulationIO:
    def __init__(self, config):
        self.config = config
        self.output_dir = 'output'
        
    def save_state(self, step, time, flow_solver, phase_field):
        """シミュレーション状態の保存"""
        filename = f'{self.output_dir}/state_{step:06d}.h5'
        with h5py.File(filename, 'w') as f:
            # メタデータの保存
            f.attrs['time'] = time
            f.attrs['step'] = step
            f.attrs['date'] = str(datetime.now())
            
            # 物理量の保存
            f.create_dataset('u', data=flow_solver.u)
            f.create_dataset('v', data=flow_solver.v)
            f.create_dataset('w', data=flow_solver.w)
            f.create_dataset('p', data=flow_solver.p)
            f.create_dataset('phi', data=phase_field.phi)
            
            # 設定の保存
            config_group = f.create_group('config')
            for key, value in vars(self.config).items():
                if isinstance(value, (int, float, str, bool)):
                    config_group.attrs[key] = value
    
    def load_state(self, filename, flow_solver, phase_field):
        """シミュレーション状態の読み込み"""
        with h5py.File(filename, 'r') as f:
            # メタデータの読み込み
            time = f.attrs['time']
            step = f.attrs['step']
            
            # 物理量の読み込み
            flow_solver.u = f['u'][:]
            flow_solver.v = f['v'][:]
            flow_solver.w = f['w'][:]
            flow_solver.p = f['p'][:]
            phase_field.phi = f['phi'][:]
            
        return time, step

    def save_for_visualization(self, step, time, flow_solver, phase_field):
        """可視化用データの保存（軽量版）"""
        filename = f'{self.output_dir}/viz_{step:06d}.h5'
        with h5py.File(filename, 'w') as f:
            # メタデータ
            f.attrs['time'] = time
            f.attrs['step'] = step
            
            # 速度場のノルム
            vel_mag = np.sqrt(
                flow_solver.u**2 +
                flow_solver.v**2 +
                flow_solver.w**2
            )
            f.create_dataset('velocity_magnitude', data=vel_mag)
            f.create_dataset('phi', data=phase_field.phi)
