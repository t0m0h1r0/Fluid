import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class CompactSolver:
    def __init__(self, config):
        self.config = config
        self._setup_matrices()
    
    def _setup_matrices(self):
        """コンパクトスキームの係数行列を設定"""
        N = self.config.Nx
        h = self.config.dx
        
        # 4次精度コンパクトスキーム
        alpha = 1/4  # コンパクトスキームのパラメータ
        a = (3/2) / h  # 右辺の係数
        
        # 主対角と副対角の設定
        main_diag = np.ones(N)
        off_diag = alpha * np.ones(N-1)
        
        # 周期境界条件の実装
        self.A = diags([main_diag, off_diag, off_diag],
                      [0, 1, -1], format='csr')
        self.A_periodic = self.A.copy()
        self.A_periodic[0, -1] = alpha
        self.A_periodic[-1, 0] = alpha
        
        # 右辺の差分作用素
        self.B = diags([0, a, -a],
                      [0, 1, -1], format='csr')
        self.B_periodic = self.B.copy()
        self.B_periodic[0, -1] = -a
        self.B_periodic[-1, 0] = a
    
    def solve_pressure(self, p, rhs):
        """圧力のポアソン方程式を解く"""
        for i in range(self.config.Ny):
            for j in range(self.config.Nz):
                p[:, i, j] = spsolve(self.A_periodic, rhs[:, i, j])
        return p
    
    def derivative(self, f, axis=0, boundary='periodic'):
        """指定方向の空間微分を計算"""
        if boundary == 'periodic':
            A = self.A_periodic
            B = self.B_periodic
        else:
            A = self.A
            B = self.B
        
        result = np.zeros_like(f)
        if axis == 0:
            for i in range(self.config.Ny):
                for j in range(self.config.Nz):
                    rhs = B @ f[:, i, j]
                    result[:, i, j] = spsolve(A, rhs)
        elif axis == 1:
            for i in range(self.config.Nx):
                for j in range(self.config.Nz):
                    rhs = B @ f[i, :, j]
                    result[i, :, j] = spsolve(A, rhs)
        elif axis == 2:
            for i in range(self.config.Nx):
                for j in range(self.config.Ny):
                    rhs = B @ f[i, j, :]
                    result[i, j, :] = spsolve(A, rhs)
        return result
