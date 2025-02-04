import numpy as np
from typing import Dict, Any, Tuple, Optional
from ..solvers.ccd_solver import CCDSolver

class NavierStokesSolver(CCDSolver):
    """
    3D Navier-Stokes equations solver using CCD scheme with ADI method
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NS solver
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing:
            - Grid parameters (nx, ny, nz, dx, dy, dz)
            - Solver parameters (order, use_filter)
            - Physical parameters (gravity)
            - Time stepping parameters (dt)
            - Fluid properties (density, viscosity)
        """
        super().__init__(config)
        
        # Physical parameters
        self.gravity = config.get('gravity', 9.81)
        
        # Initialize fluid properties fields
        self.rho = np.zeros((self.nx, self.ny, self.nz))  # Density field
        self.mu = np.zeros((self.nx, self.ny, self.nz))   # Viscosity field
        self.phase = np.zeros((self.nx, self.ny, self.nz))  # Phase field
        
        # Initialize fields needed for pressure projection
        self.pressure_correction = np.zeros((self.nx, self.ny, self.nz))
        self.u_star = np.zeros((self.nx, self.ny, self.nz))
        self.v_star = np.zeros((self.nx, self.ny, self.nz))
        self.w_star = np.zeros((self.nx, self.ny, self.nz))
        
        # Initialize ADI solver matrices
        self._init_adi_matrices()
    
    def apply_boundary_condition(self) -> None:
        """Apply boundary conditions to all fields"""
        if self.boundary_manager is not None:
            self.u, self.v, self.w, self.p = self.boundary_manager.apply_ns_boundary_conditions(
                self.u, self.v, self.w, self.p
            )
        else:
            # Default Neumann boundary conditions if no boundary manager is set
            # x-direction
            self.u[0, :, :] = self.u[1, :, :]
            self.u[-1, :, :] = self.u[-2, :, :]
            self.v[0, :, :] = self.v[1, :, :]
            self.v[-1, :, :] = self.v[-2, :, :]
            self.w[0, :, :] = self.w[1, :, :]
            self.w[-1, :, :] = self.w[-2, :, :]
            
            # y-direction
            self.u[:, 0, :] = self.u[:, 1, :]
            self.u[:, -1, :] = self.u[:, -2, :]
            self.v[:, 0, :] = self.v[:, 1, :]
            self.v[:, -1, :] = self.v[:, -2, :]
            self.w[:, 0, :] = self.w[:, 1, :]
            self.w[:, -1, :] = self.w[:, -2, :]
            
            # z-direction
            self.u[:, :, 0] = self.u[:, :, 1]
            self.u[:, :, -1] = self.u[:, :, -2]
            self.v[:, :, 0] = self.v[:, :, 1]
            self.v[:, :, -1] = self.v[:, :, -2]
            self.w[:, :, 0] = self.w[:, :, 1]
            self.w[:, :, -1] = self.w[:, :, -2]
            
            # Pressure boundary conditions
            self.p[0, :, :] = self.p[1, :, :]
            self.p[-1, :, :] = self.p[-2, :, :]
            self.p[:, 0, :] = self.p[:, 1, :]
            self.p[:, -1, :] = self.p[:, -2, :]
            self.p[:, :, 0] = self.p[:, :, 1]
            self.p[:, :, -1] = self.p[:, :, -2]
    
    def _init_adi_matrices(self) -> None:
        """Initialize matrices for ADI method"""
        # ADI coefficients for x-direction
        self.Ax_adi = np.zeros((3, self.nx))
        self.Ax_adi[1, :] = 1.0 + 2.0/(self.dx**2)  # main diagonal
        self.Ax_adi[0, 1:] = -1.0/(self.dx**2)  # upper diagonal
        self.Ax_adi[2, :-1] = -1.0/(self.dx**2)  # lower diagonal
        
        # ADI coefficients for y-direction
        self.Ay_adi = np.zeros((3, self.ny))
        self.Ay_adi[1, :] = 1.0 + 2.0/(self.dy**2)
        self.Ay_adi[0, 1:] = -1.0/(self.dy**2)
        self.Ay_adi[2, :-1] = -1.0/(self.dy**2)
        
        # ADI coefficients for z-direction
        self.Az_adi = np.zeros((3, self.nz))
        self.Az_adi[1, :] = 1.0 + 2.0/(self.dz**2)
        self.Az_adi[0, 1:] = -1.0/(self.dz**2)
        self.Az_adi[2, :-1] = -1.0/(self.dz**2)
    
    def compute_convective_terms(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convective terms of NS equations"""
        # Compute velocity derivatives
        dudx = self.compute_derivatives(self.u, 'x')
        dudy = self.compute_derivatives(self.u, 'y')
        dudz = self.compute_derivatives(self.u, 'z')
        dvdx = self.compute_derivatives(self.v, 'x')
        dvdy = self.compute_derivatives(self.v, 'y')
        dvdz = self.compute_derivatives(self.v, 'z')
        dwdx = self.compute_derivatives(self.w, 'x')
        dwdy = self.compute_derivatives(self.w, 'y')
        dwdz = self.compute_derivatives(self.w, 'z')
        
        # Compute convective terms
        conv_u = -(self.u * dudx + self.v * dudy + self.w * dudz)
        conv_v = -(self.u * dvdx + self.v * dvdy + self.w * dvdz)
        conv_w = -(self.u * dwdx + self.v * dwdy + self.w * dwdz)
        
        return conv_u, conv_v, conv_w
    
    def compute_diffusive_terms(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute diffusive terms of NS equations"""
        # Second derivatives
        d2udx2 = self.compute_second_derivatives(self.u, 'x')
        d2udy2 = self.compute_second_derivatives(self.u, 'y')
        d2udz2 = self.compute_second_derivatives(self.u, 'z')
        d2vdx2 = self.compute_second_derivatives(self.v, 'x')
        d2vdy2 = self.compute_second_derivatives(self.v, 'y')
        d2vdz2 = self.compute_second_derivatives(self.v, 'z')
        d2wdx2 = self.compute_second_derivatives(self.w, 'x')
        d2wdy2 = self.compute_second_derivatives(self.w, 'y')
        d2wdz2 = self.compute_second_derivatives(self.w, 'z')
        
        # Compute diffusive terms with variable viscosity
        visc_u = (self.mu/self.rho) * (d2udx2 + d2udy2 + d2udz2)
        visc_v = (self.mu/self.rho) * (d2vdx2 + d2vdy2 + d2vdz2)
        visc_w = (self.mu/self.rho) * (d2wdx2 + d2wdy2 + d2wdz2)
        
        return visc_u, visc_v, visc_w
    
    def compute_gravity_terms(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gravity terms"""
        # Reference density (water)
        rho_ref = self.rho[0, 0, 0]  # Assuming bottom is water
        
        # Only z-direction gravity with buoyancy
        gx = np.zeros_like(self.u)
        gy = np.zeros_like(self.v)
        
        # Compute buoyancy term safely
        gz = -self.gravity * np.ones_like(self.w)
        rho_ratio = np.divide(
            self.rho - rho_ref,
            self.rho,
            out=np.zeros_like(self.rho),
            where=self.rho > 1e-6
        )
        gz *= rho_ratio
        
        return gx, gy, gz
    
    def solve_pressure_poisson(self) -> None:
        """Solve pressure Poisson equation using ADI method"""
        # Compute divergence of intermediate velocity
        dudx = self.compute_derivatives(self.u_star, 'x')
        dvdy = self.compute_derivatives(self.v_star, 'y')
        dwdz = self.compute_derivatives(self.w_star, 'z')
        divergence = dudx + dvdy + dwdz
        
        # RHS for Poisson equation
        rhs = divergence / self.dt
        
        # Solve using ADI method
        self.pressure_correction = self._adi_poisson_solver(rhs)
        
        # Update pressure
        self.p += self.pressure_correction
    
    def _adi_poisson_solver(self, rhs: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation using ADI method
        
        Parameters
        ----------
        rhs : ndarray
            Right-hand side of Poisson equation
        """
        # Intermediate arrays
        phi = np.zeros_like(rhs)
        psi = np.zeros_like(rhs)
        pressure = np.zeros_like(rhs)
        
        # X-direction sweep
        for j in range(self.ny):
            for k in range(self.nz):
                phi[:, j, k] = self._solve_tridiagonal(
                    self.Ax_adi, rhs[:, j, k])
        
        # Y-direction sweep
        for i in range(self.nx):
            for k in range(self.nz):
                psi[i, :, k] = self._solve_tridiagonal(
                    self.Ay_adi, phi[i, :, k])
        
        # Z-direction sweep
        for i in range(self.nx):
            for j in range(self.ny):
                pressure[i, j, :] = self._solve_tridiagonal(
                    self.Az_adi, psi[i, j, :])
        
        return pressure
    
    def _solve_tridiagonal(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve tridiagonal system using Thomas algorithm"""
        n = len(b)
        x = np.zeros_like(b)
        
        # Forward elimination
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)
        
        c_prime[0] = A[0, 1] / A[1, 0]
        d_prime[0] = b[0] / A[1, 0]
        
        for i in range(1, n-1):
            denominator = A[1, i] - A[2, i-1] * c_prime[i-1]
            c_prime[i] = A[0, i+1] / denominator
            d_prime[i] = (b[i] - A[2, i-1] * d_prime[i-1]) / denominator
        
        d_prime[n-1] = (b[n-1] - A[2, n-2] * d_prime[n-2]) / \
                      (A[1, n-1] - A[2, n-2] * c_prime[n-2])
        
        # Back substitution
        x[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x
    
    def compute_second_derivatives(self, field: np.ndarray, 
                                 direction: str) -> np.ndarray:
        """Compute second derivatives using CCD scheme"""
        # First compute first derivative
        first_deriv = self.compute_derivatives(field, direction)
        # Then compute derivative of the first derivative
        return self.compute_derivatives(first_deriv, direction)
    
    def step(self) -> None:
        """Perform one time step using projection method"""
        # Store previous velocity
        u_prev = self.u.copy()
        v_prev = self.v.copy()
        w_prev = self.w.copy()
        
        # Compute all terms
        conv_u, conv_v, conv_w = self.compute_convective_terms()
        visc_u, visc_v, visc_w = self.compute_diffusive_terms()
        gx, gy, gz = self.compute_gravity_terms()
        
        # Predictor step (compute intermediate velocity)
        self.u_star = u_prev + self.dt * (conv_u + visc_u + gx)
        self.v_star = v_prev + self.dt * (conv_v + visc_v + gy)
        self.w_star = w_prev + self.dt * (conv_w + visc_w + gz)
        
        # Project velocity to satisfy incompressibility
        self.solve_pressure_poisson()
        
        # Compute pressure gradients
        dpdx = self.compute_derivatives(self.pressure_correction, 'x')
        dpdy = self.compute_derivatives(self.pressure_correction, 'y')
        dpdz = self.compute_derivatives(self.pressure_correction, 'z')
        
        # Corrector step
        self.u = self.u_star - self.dt * dpdx / self.rho
        self.v = self.v_star - self.dt * dpdy / self.rho
        self.w = self.w_star - self.dt * dpdz / self.rho
        
        # Apply boundary conditions
        self.apply_boundary_condition()
        
        # Apply filter if enabled
        if self.use_filter:
            self.u = self.apply_filter(self.u)
            self.v = self.apply_filter(self.v)
            self.w = self.apply_filter(self.w)
            self.p = self.apply_filter(self.p)
        
        # Update time
        self.t += self.dt

    def get_cfl_number(self) -> float:
        """
        Compute CFL number
        
        Returns
        -------
        float
            Current CFL number
        """
        max_vel = max(
            np.max(np.abs(self.u)) / self.dx,
            np.max(np.abs(self.v)) / self.dy,
            np.max(np.abs(self.w)) / self.dz
        )
        return max_vel * self.dt
    
    def adjust_timestep(self, target_cfl: float = 0.5) -> None:
        """
        Adjust timestep to maintain target CFL number
        
        Parameters
        ----------
        target_cfl : float
            Target CFL number
        """
        current_cfl = self.get_cfl_number()
        if current_cfl > 0:
            self.dt *= target_cfl / current_cfl
    
    def solve(self, end_time: float) -> None:
        """
        Solve until specified end time
        
        Parameters
        ----------
        end_time : float
            Time to solve until
        """
        n_steps = int((end_time - self.t) / self.dt)
        
        for step in range(n_steps):
            self.step()
            
            # Print progress and diagnostics every 100 steps
            if step % 100 == 0:
                max_vel = max(
                    np.max(np.abs(self.u)),
                    np.max(np.abs(self.v)),
                    np.max(np.abs(self.w))
                )
                print(f"Step {step}/{n_steps}")
                print(f"Time: {self.t:.3f}")
                print(f"Max velocity: {max_vel:.3e}")
                print(f"CFL number: {self.get_cfl_number():.3f}")
                print("-" * 40)
            
            # Adjust timestep if needed
            self.adjust_timestep()
    
    def get_diagnostics(self) -> Dict[str, float]:
        """
        Compute diagnostic quantities
        
        Returns
        -------
        Dict[str, float]
            Dictionary of diagnostic quantities
        """
        # Compute kinetic energy
        ke = 0.5 * np.sum(self.rho * (self.u**2 + self.v**2 + self.w**2))
        
        # Compute maximum velocity magnitude
        max_vel = np.max(np.sqrt(self.u**2 + self.v**2 + self.w**2))
        
        # Compute enstrophy
        vorticity_x = self.compute_derivatives(self.w, 'y') - self.compute_derivatives(self.v, 'z')
        vorticity_y = self.compute_derivatives(self.u, 'z') - self.compute_derivatives(self.w, 'x')
        vorticity_z = self.compute_derivatives(self.v, 'x') - self.compute_derivatives(self.u, 'y')
        enstrophy = 0.5 * np.sum(vorticity_x**2 + vorticity_y**2 + vorticity_z**2)
        
        # Compute divergence error
        dudx = self.compute_derivatives(self.u, 'x')
        dvdy = self.compute_derivatives(self.v, 'y')
        dwdz = self.compute_derivatives(self.w, 'z')
        div_error = np.max(np.abs(dudx + dvdy + dwdz))
        
        return {
            'time': self.t,
            'dt': self.dt,
            'kinetic_energy': ke,
            'max_velocity': max_vel,
            'enstrophy': enstrophy,
            'divergence_error': div_error,
            'cfl_number': self.get_cfl_number()
        }
    
    def write_fields(self, filename: str) -> None:
        """
        Write field data to file
        
        Parameters
        ----------
        filename : str
            Name of file to write to
        """
        data = {
            'u': self.u,
            'v': self.v,
            'w': self.w,
            'p': self.p,
            'rho': self.rho,
            'mu': self.mu,
            'phase': self.phase,
            'time': self.t,
            'dt': self.dt,
            'nx': self.nx,
            'ny': self.ny,
            'nz': self.nz,
            'dx': self.dx,
            'dy': self.dy,
            'dz': self.dz
        }
        np.savez_compressed(filename, **data)
    
    @classmethod
    def load_fields(cls, filename: str, config: Dict[str, Any]):
        """
        Load field data from file
        
        Parameters
        ----------
        filename : str
            Name of file to load from
        config : Dict[str, Any]
            Configuration dictionary for solver initialization
            
        Returns
        -------
        NavierStokesSolver
            Solver instance with loaded fields
        """
        data = np.load(filename)
        solver = cls(config)
        
        # Load fields
        solver.u = data['u']
        solver.v = data['v']
        solver.w = data['w']
        solver.p = data['p']
        solver.rho = data['rho']
        solver.mu = data['mu']
        solver.phase = data['phase']
        solver.t = float(data['time'])
        solver.dt = float(data['dt'])
        
        return solver