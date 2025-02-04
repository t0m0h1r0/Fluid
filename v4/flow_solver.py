import numpy as np

class FlowSolver:
    def __init__(self, config, phase_field, compact_solver):
        self.config = config
        self.phase = phase_field
        self.solver = compact_solver
        
        # 速度場の初期化
        self.u = np.zeros((config.Nx, config.Ny, config.Nz))
        self.v = np.zeros((config.Nx, config.Ny, config.Nz))
        self.w = np.zeros((config.Nx, config.Ny, config.Nz))
        self.p = np.zeros((config.Nx, config.Ny, config.Nz))
        
    def get_density(self):
        """密度場の計算"""
        H = self.phase.heaviside(self.phase.phi)
        return (1 - H) * self.config.rho_water + H * self.config.rho_nitrogen
    
    def get_viscosity(self):
        """粘性場の計算"""
        H = self.phase.heaviside(self.phase.phi)
        return (1 - H) * self.config.mu_water + H * self.config.mu_nitrogen
    
    def rhs_momentum(self, u, v, w, rho, mu):
        """運動方程式の右辺を計算"""
        cfg = self.config
        dx, dy, dz = cfg.dx, cfg.dy, cfg.dz
        
        # 対流項の計算
        conv_x = (
            self.solver.derivative(u * u, axis=0) +
            self.solver.derivative(u * v, axis=1) +
            self.solver.derivative(u * w, axis=2)
        )
        conv_y = (
            self.solver.derivative(v * u, axis=0) +
            self.solver.derivative(v * v, axis=1) +
            self.solver.derivative(v * w, axis=2)
        )
        conv_z = (
            self.solver.derivative(w * u, axis=0) +
            self.solver.derivative(w * v, axis=1) +
            self.solver.derivative(w * w, axis=2)
        )
        
        # 粘性項の計算
        visc_x = (
            self.solver.derivative(mu * self.solver.derivative(u, axis=0), axis=0) +
            self.solver.derivative(mu * self.solver.derivative(u, axis=1), axis=1) +
            self.solver.derivative(mu * self.solver.derivative(u, axis=2), axis=2)
        )
        visc_y = (
            self.solver.derivative(mu * self.solver.derivative(v, axis=0), axis=0) +
            self.solver.derivative(mu * self.solver.derivative(v, axis=1), axis=1) +
            self.solver.derivative(mu * self.solver.derivative(v, axis=2), axis=2)
        )
        visc_z = (
            self.solver.derivative(mu * self.solver.derivative(w, axis=0), axis=0) +
            self.solver.derivative(mu * self.solver.derivative(w, axis=1), axis=1) +
            self.solver.derivative(mu * self.solver.derivative(w, axis=2), axis=2)
        )
        
        # 重力項の計算（z方向のみ）
        grav_x = 0
        grav_y = 0
        grav_z = -self.config.g
        
        return (
            np.array([-conv_x/rho + visc_x/rho + grav_x,
                     -conv_y/rho + visc_y/rho + grav_y,
                     -conv_z/rho + visc_z/rho + grav_z])
        )
    
    def runge_kutta_step(self):
        """4次のルンゲ・クッタ法による時間発展"""
        dt = self.config.dt
        rho = self.get_density()
        mu = self.get_viscosity()
        
        # RK4のステージ
        k1 = dt * self.rhs_momentum(
            self.u, self.v, self.w, rho, mu
        )
        
        k2 = dt * self.rhs_momentum(
            self.u + 0.5*k1[0],
            self.v + 0.5*k1[1],
            self.w + 0.5*k1[2],
            rho, mu
        )
        
        k3 = dt * self.rhs_momentum(
            self.u + 0.5*k2[0],
            self.v + 0.5*k2[1],
            self.w + 0.5*k2[2],
            rho, mu
        )
        
        k4 = dt * self.rhs_momentum(
            self.u + k3[0],
            self.v + k3[1],
            self.w + k3[2],
            rho, mu
        )
        
        # 速度場の更新
        self.u += (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
        self.v += (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
        self.w += (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
        
        # 圧力補正
        self.pressure_correction()
    
    def pressure_correction(self):
        """圧力補正による非圧縮性の保証"""
        # 発散場の計算
        div_u = (
            self.solver.derivative(self.u, axis=0) +
            self.solver.derivative(self.v, axis=1) +
            self.solver.derivative(self.w, axis=2)
        )
        
        rho = self.get_density()
        rhs = rho * div_u / self.config.dt
        
        # 圧力ポアソン方程式を解く
        self.p = self.solver.solve_pressure(self.p, rhs)
        
        # 速度場の補正
        self.u -= self.config.dt * self.solver.derivative(self.p, axis=0) / rho
        self.v -= self.config.dt * self.solver.derivative(self.p, axis=1) / rho
        self.w -= self.config.dt * self.solver.derivative(self.p, axis=2) / rho
