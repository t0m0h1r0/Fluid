from typing import Optional, Dict, Any, Union, Tuple
import jax
import jax.numpy as jnp
from functools import partial

from ..config import PoissonSolverConfig
from .base import PoissonSolverBase
from core.field import ScalarField

class PoissonMultigridSolver(PoissonSolverBase):
    def __init__(self, config: Optional[PoissonSolverConfig] = None, cycle_type: str = "V", num_levels: int = 3):
        super().__init__(config)
        self.cycle_type = cycle_type
        self.num_levels = num_levels
        jax.config.update('jax_enable_x64', True)
        self._init_multigrid_operators()

    def _init_multigrid_operators(self):
        @partial(jax.jit, static_argnums=(1,))
        def restrict(fine_grid: jnp.ndarray, shape: Tuple[int, int, int]) -> jnp.ndarray:
            coarse_shape = tuple(s // 2 for s in shape)
            return jax.vmap(lambda i, j, k: jnp.mean(fine_grid[2*i:2*i+2, 2*j:2*j+2, 2*k:2*k+2]),
                            (0, 0, 0))(jnp.arange(coarse_shape[0]), jnp.arange(coarse_shape[1]), jnp.arange(coarse_shape[2]))
        
        @partial(jax.jit, static_argnums=(1,))
        def prolongate(coarse_grid: jnp.ndarray, fine_shape: Tuple[int, int, int]) -> jnp.ndarray:
            fine_grid = jnp.zeros(fine_shape)
            fine_grid = fine_grid.at[::2, ::2, ::2].set(coarse_grid)
            fine_grid = jax.lax.map(lambda i: (fine_grid.at[i::2].set((fine_grid[i::2-1] + fine_grid[i::2+1]) / 2)), jnp.array([1]))
            return fine_grid

        @partial(jax.jit, static_argnums=(3,))
        def smooth(solution: jnp.ndarray, rhs: jnp.ndarray, dx: jnp.ndarray, is_coarse_grid: bool) -> jnp.ndarray:
            num_iterations = 50 if is_coarse_grid else 2
            dx2 = dx * dx
            
            def sor_iteration(_, u):
                neighbors = (
                    (u[:-2, 1:-1, 1:-1] + u[2:, 1:-1, 1:-1]) / dx2[0] +
                    (u[1:-1, :-2, 1:-1] + u[1:-1, 2:, 1:-1]) / dx2[1] +
                    (u[1:-1, 1:-1, :-2] + u[1:-1, 1:-1, 2:]) / dx2[2]
                )
                coeff = 2.0 * (1 / dx2[0] + 1 / dx2[1] + 1 / dx2[2])
                return u.at[1:-1, 1:-1, 1:-1].set((1 - 1.5) * u[1:-1, 1:-1, 1:-1] + (1.5 / coeff) * (neighbors - rhs[1:-1, 1:-1, 1:-1]))
            
            return jax.lax.fori_loop(0, num_iterations, sor_iteration, solution)

        self.restrict = restrict
        self.prolongate = prolongate
        self.smooth = smooth
        self.laplacian_operator = jax.jit(self.laplacian_operator)

    def solve(self, rhs: Union[jnp.ndarray, ScalarField], initial_guess: Optional[Union[jnp.ndarray, ScalarField]] = None) -> jnp.ndarray:
        rhs_array, initial_array = self.validate_input(rhs, initial_guess)
        solution = initial_array if initial_array is not None else jnp.zeros_like(rhs_array)
        dx = jnp.array(self.config.dx)

        def multigrid_cycle(solution, rhs, level):
            if level == self.num_levels - 1:
                return self.smooth(solution, rhs, dx, is_coarse_grid=True)
            solution = self.smooth(solution, rhs, dx, is_coarse_grid=False)
            residual = rhs - self.laplacian_operator(solution)
            coarse_residual = self.restrict(residual, residual.shape)
            coarse_solution = jnp.zeros_like(coarse_residual)
            coarse_solution = multigrid_cycle(coarse_solution, coarse_residual, level + 1)
            correction = self.prolongate(coarse_solution, solution.shape)
            solution = solution + correction
            return self.smooth(solution, rhs, dx, is_coarse_grid=False)

        def solver_iteration(state):
            solution, best_solution, best_residual = state
            solution = multigrid_cycle(solution, rhs_array, 0)
            residual = jnp.max(jnp.abs(rhs_array - self.laplacian_operator(solution)))
            new_best_solution = jax.lax.cond(residual < best_residual, lambda _: solution, lambda _: best_solution, None)
            return solution, new_best_solution, jnp.minimum(residual, best_residual)

        initial_state = (solution, solution, jnp.inf)
        final_state = jax.lax.fori_loop(0, self.config.max_iterations, lambda _, state: solver_iteration(state), initial_state)
        solution, best_solution, final_residual = final_state
        self._iteration_count = self.config.max_iterations
        self._converged = final_residual < self.config.tolerance
        self._error_history = [float(final_residual)]
        return best_solution

    def get_diagnostics(self) -> Dict[str, Any]:
        diag = super().get_diagnostics()
        diag.update({
            "solver_type": "Multigrid",
            "cycle_type": self.cycle_type,
            "num_levels": self.num_levels,
            "final_residual": self._error_history[-1] if self._error_history else None,
        })
        return diag
