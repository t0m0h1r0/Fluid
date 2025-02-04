import numpy as np
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve

class CompactSolver:
    def __init__(self, config):
        self.config = config
        self._setup_matrices()
    
    def _setup_matrices(self):
        """コンパクトスキームの係数行列を設定"""
        Nx, Ny, Nz = self.config.Nx, self.config.Ny, self.config.Nz
        dx = self.config.dx
        
        # 4次精度コンパクトスキーム
        alpha = 1/4
        a = (3/2) / dx
        
        # 各方向の行列を初期化
        self.A = []
        self.A_periodic = []
        self.B = []
        self.B_periodic = []
        
        for N in [Nx, Ny, Nz]:
            # 主対角と副対角
            main_diag = np.ones(N)
            off_diag = alpha * np.ones(N-1)
            
            # 通常の行列
            A = lil_matrix((N, N))
            A.setdiag(main_diag)
            A.setdiag(off_diag, k=1)
            A.setdiag(off_diag, k=-1)
            
            # 周期境界条件の行列
            A_periodic = A.copy()
            A_periodic[0, -1] = alpha
            A_periodic[-1, 0] = alpha
            
            # 右辺の差分作用素
            B = lil_matrix((N, N))
            B.setdiag(np.zeros(N))
            B.setdiag(a * np.ones(N-1), k=1)
            B.setdiag(-a * np.ones(N-1), k=-1)
            
            # 周期境界条件
            B_periodic = B.copy()
            B_periodic[0, -1] = -a
            B_periodic[-1, 0] = a
            
            # CSR形式に変換して保存
            self.A.append(A.tocsr())
            self.A_periodic.append(A_periodic.tocsr())
            self.B.append(B.tocsr())
            self.B_periodic.append(B_periodic.tocsr())
    
    def solve_pressure(self, p, rhs):
        """圧力のポアソン方程式を解く"""
        Nx, Ny, Nz = self.config.Nx, self.config.Ny, self.config.Nz
        result = np.zeros_like(p)
        
        total = Ny * Nz
        count = 0
        
        for i in range(Ny):
            for j in range(Nz):
                result[:, i, j] = spsolve(self.A_periodic[0], rhs[:, i, j])
                count += 1
                if count % 10 == 0:
                    print(f"Solving Poisson equation: {count}/{total} ({count/total*100:.1f}%)")
        
        return result
    
    def derivative(self, f, axis=0, boundary='periodic'):
        """指定方向の空間微分を計算"""
        if boundary == 'periodic':
            A = self.A_periodic[axis]
            B = self.B_periodic[axis]
        else:
            A = self.A[axis]
            B = self.B[axis]
        
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