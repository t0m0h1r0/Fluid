import numpy as np

class PhaseField:
    def __init__(self, config):
        self.config = config
        self.phi = np.zeros((config.Nx, config.Ny, config.Nz))
        self.initialize_phase_field()
    
    def initialize_phase_field(self):
        """初期条件の設定"""
        cfg = self.config
        x, y, z = np.meshgrid(cfg.x, cfg.y, cfg.z, indexing='ij')
        
        # 球体の設定
        r = np.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.4)**2)
        self.phi[r <= 0.2] = 1.0
        
        # z > 1.5 の領域を窒素に
        self.phi[..., z[0,0,:] > 1.5] = 1.0
    
    def heaviside(self, phi):
        """ヘヴィサイド関数の近似"""
        epsilon = 1.0e-2
        return 0.5 * (1.0 + np.tanh(phi / epsilon))
