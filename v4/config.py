import numpy as np

class SimulationConfig:
    def __init__(self):
        # 計算領域の設定
        self.Lx, self.Ly, self.Lz = 1.0, 1.0, 2.0
        self.Nx, self.Ny, self.Nz = 16, 16, 32
        
        # 物理パラメータ
        self.rho_water = 1000.0  # 水の密度 [kg/m^3]
        self.rho_nitrogen = 1.225  # 窒素の密度 [kg/m^3]
        self.mu_water = 1.0e-3  # 水の粘性係数 [Pa・s]
        self.mu_nitrogen = 1.79e-5  # 窒素の粘性係数 [Pa・s]
        self.g = 9.81  # 重力加速度 [m/s^2]
        
        # 時間発展の設定
        self.dt = 0.001  # 時間刻み
        self.save_interval = 0.1  # 保存間隔
        
        # グリッドの生成
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.z = np.linspace(0, self.Lz, self.Nz)
