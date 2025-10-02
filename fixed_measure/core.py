"""
Core solver implementation with maximum performance optimization.

The design separates static (compilation-dependent) and dynamic (data-dependent)
components to avoid recompilation in optimization loops.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import List, Tuple, Optional


# ============================================================================
# Static JIT-compiled functions
# These compile once per unique combination of static arguments (d, eps, etc.)
# ============================================================================

@jax.jit
def _precompute_transforms(F: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute inverse transformations and Jacobian determinants.
    JIT-compiled for efficiency.

    Args:
        F: Transformation matrices (n, 3, 3)

    Returns:
        F_inv: Inverse matrices (n, 3, 3)
        jac_dets: Absolute Jacobian determinants (n,)
    """
    F_inv = jnp.linalg.inv(F)
    A = F[:, :2, :2]
    jac_dets = jnp.abs(jnp.linalg.det(A))
    return F_inv, jac_dets


@partial(jax.jit, static_argnames=('d',))
def _compute_transformed_grid(base_grid: jnp.ndarray, f_inv: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Apply inverse transformation to base grid and convert to pixel coordinates.

    Args:
        base_grid: Homogeneous coordinates grid (d*d, 3)
        f_inv: Inverse transformation matrix (3, 3)
        d: Grid dimension

    Returns:
        Pixel coordinates for map_coordinates (2, d, d)
    """
    # Apply inverse transformation
    transformed = f_inv @ base_grid.T

    # De-homogenize
    x_norm = transformed[0] / transformed[2]
    y_norm = transformed[1] / transformed[2]

    # Convert to pixel indices [0, d-1]
    x_pixel = x_norm * (d - 1.0)
    y_pixel = y_norm * (d - 1.0)

    # Stack as (y, x) for map_coordinates
    coords = jnp.stack([
        y_pixel.reshape(d, d),
        x_pixel.reshape(d, d)
    ], axis=0)

    return coords


@partial(jax.jit, static_argnames=('d',))
def _single_pushforward(mu: jnp.ndarray, base_grid: jnp.ndarray, f_inv: jnp.ndarray,
                        jac_det: float, prob: float, d: int) -> jnp.ndarray:
    """
    Compute push-forward for a single transformation.

    Args:
        mu: Current measure (d, d)
        base_grid: Base grid in homogeneous coordinates (d*d, 3)
        f_inv: Inverse transformation matrix (3, 3)
        jac_det: Jacobian determinant (scalar)
        prob: Probability weight (scalar)
        d: Grid dimension

    Returns:
        Weighted pushed measure (d, d)
    """
    coords = _compute_transformed_grid(base_grid, f_inv, d)

    pushed = jax.scipy.ndimage.map_coordinates(
        mu, coords, order=1, mode='constant', cval=0.0
    )

    # Apply Jacobian correction
    pushed = jnp.where(jac_det > 1e-9, pushed / jac_det, pushed)

    return prob * pushed


@partial(jax.jit, static_argnames=('d',))
def _push_forward_step(mu: jnp.ndarray, base_grid: jnp.ndarray, F_inv: jnp.ndarray,
                       jac_dets: jnp.ndarray, p: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Compute full push-forward: Σ p_i * (f_i)_# μ

    Args:
        mu: Current measure (d, d)
        base_grid: Base grid (d*d, 3)
        F_inv: Inverse transformations (n, 3, 3)
        jac_dets: Jacobian determinants (n,)
        p: Probability weights (n,)
        d: Grid dimension

    Returns:
        Next measure (d, d)
    """
    # Vectorize over all transformations
    pushed_components = jax.vmap(
        _single_pushforward, in_axes=(None, None, 0, 0, 0, None)
    )(mu, base_grid, F_inv, jac_dets, p, d)

    mu_next = jnp.sum(pushed_components, axis=0)
    return mu_next / mu_next.sum()


@partial(jax.jit, static_argnames=('d', 'eps', 'max_iterations', 'min_iterations'))
def _solve_fixed_measure(F_inv: jnp.ndarray, jac_dets: jnp.ndarray, p: jnp.ndarray,
                         mu_initial: jnp.ndarray, base_grid: jnp.ndarray,
                         d: int, eps: float, max_iterations: int, min_iterations: int
                         ) -> Tuple[jnp.ndarray, int, float]:
    """
    Solve for fixed measure using iterative push-forward.

    This is the main computation kernel. It compiles once per unique combination
    of (d, eps, max_iterations, min_iterations) and can be reused with different
    F_inv, jac_dets, p, mu_initial values without recompilation.

    Args:
        F_inv: Inverse transformations (n, 3, 3)
        jac_dets: Jacobian determinants (n,)
        p: Probability weights (n,)
        mu_initial: Initial measure (d, d)
        base_grid: Base grid (d*d, 3)
        d: Grid dimension
        eps: Convergence threshold
        max_iterations: Maximum iterations
        min_iterations: Minimum iterations before checking convergence

    Returns:
        final_mu: Converged measure (d, d)
        iters: Number of iterations performed
        w_inf: Final Wasserstein-infinity distance
    """
    def cond_fun(state):
        _, iteration, w_inf = state
        converged = (iteration >= min_iterations) & (w_inf < eps)
        not_done = iteration < max_iterations
        return not_done & (~converged)

    def body_fun(state):
        mu, iteration, _ = state
        mu_next = _push_forward_step(mu, base_grid, F_inv, jac_dets, p, d)
        w_inf = jnp.max(jnp.abs(mu - mu_next))
        return mu_next, iteration + 1, w_inf

    initial_state = (mu_initial, 0, jnp.inf)
    final_mu, iters, w_inf = jax.lax.while_loop(cond_fun, body_fun, initial_state)

    return final_mu, iters, w_inf


# ============================================================================
# Solver Class
# ============================================================================

class FixedMeasureSolver:
    """
    High-performance Fixed Measure solver for Iterated Function Systems.

    This solver is designed for use in optimization loops where F and p change
    frequently. The solver compiles once and reuses compiled code across calls.

    Usage:
        # Initialize once with static parameters
        solver = FixedMeasureSolver(d=1024, eps=1e-6)

        # Optionally warm up (triggers compilation)
        solver.warmup(n_transforms=3)

        # Use in optimization loop (no recompilation!)
        for step in range(optimization_steps):
            F_new = ... # your optimization updates
            p_new = ...
            mu_fixed = solver.solve(F=F_new, p=p_new)
    """

    def __init__(self, d: int, eps: float = 1e-4,
                 max_iterations: int = 1000, min_iterations: int = 100):
        """
        Initialize solver with static parameters.

        Args:
            d: Grid dimension (must be power of 2)
            eps: Convergence threshold for Wasserstein-infinity distance
            max_iterations: Maximum iterations
            min_iterations: Minimum iterations before checking convergence
        """
        if not (d & (d - 1) == 0 and d != 0):
            raise ValueError(f"Grid dimension d={d} must be a power of 2")

        self.d = d
        self.eps = eps
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations

        # Precompute base grid (only depends on d)
        self.base_grid = self._create_base_grid()

        self._warmed_up = False

    def _create_base_grid(self) -> jnp.ndarray:
        """Create base grid in homogeneous coordinates."""
        y_coords = jnp.linspace(0.0, 1.0, self.d, dtype=jnp.float32)
        x_coords = jnp.linspace(0.0, 1.0, self.d, dtype=jnp.float32)
        grid_y, grid_x = jnp.meshgrid(y_coords, x_coords, indexing='ij')

        return jnp.stack([
            grid_x.ravel(),
            grid_y.ravel(),
            jnp.ones(self.d * self.d, dtype=jnp.float32)
        ], axis=1)

    def warmup(self, n_transforms: int = 3, verbose: bool = True) -> None:
        """
        Warm up the solver by triggering JIT compilation with dummy data.

        This is optional but recommended before running optimization loops
        to avoid compilation overhead on the first solve() call.

        Args:
            n_transforms: Number of transformations in your IFS
            verbose: Print warmup status
        """
        if verbose:
            print(f"Warming up solver (d={self.d}, n={n_transforms})...")

        # Create dummy identity transformations
        dummy_F = [jnp.eye(3, dtype=jnp.float32) for _ in range(n_transforms)]
        dummy_p = jnp.ones(n_transforms, dtype=jnp.float32) / n_transforms

        # Trigger compilation (result is discarded)
        _ = self.solve(F=dummy_F, p=dummy_p, verbose=False)

        self._warmed_up = True
        if verbose:
            print(f"✓ Solver warmed up! Subsequent calls will be fast.")

    def solve(self, F: List[jnp.ndarray], p: Optional[jnp.ndarray] = None,
              mu: Optional[jnp.ndarray] = None, verbose: bool = True) -> jnp.ndarray:
        """
        Solve for the fixed measure of the IFS defined by F and p.

        Args:
            F: List of n transformation matrices (each 3x3 JAX array)
            p: Probability vector (n,) - defaults to uniform if None
            mu: Initial measure (d, d) - defaults to uniform if None (warm-start with previous iteration's result)
            verbose: Print convergence information

        Returns:
            Fixed measure (d, d) as JAX array
        """
        # Prepare inputs
        n = len(F)

        if p is None:
            p_jax = jnp.full((n,), 1.0 / n, dtype=jnp.float32)
        else:
            p_jax = jnp.asarray(p, dtype=jnp.float32)
            if not jnp.allclose(p_jax.sum(), 1.0):
                raise ValueError("Probability vector p must sum to 1")

        F_jax = jnp.stack(F, axis=0).astype(jnp.float32)

        if mu is None:
            mu_initial = jnp.ones((self.d, self.d), dtype=jnp.float32)
            mu_initial = mu_initial / mu_initial.sum()
        else:
            mu_initial = jnp.asarray(mu, dtype=jnp.float32)
            mu_initial = mu_initial / mu_initial.sum()

        # Precompute transforms
        F_inv, jac_dets = _precompute_transforms(F_jax)

        # Solve (this is the JIT-compiled hotpath)
        final_mu, iters, w_inf = _solve_fixed_measure(
            F_inv, jac_dets, p_jax, mu_initial, self.base_grid,
            self.d, self.eps, self.max_iterations, self.min_iterations
        )

        # Wait for computation to complete
        final_mu.block_until_ready()

        # Report
        if verbose:
            if iters >= self.max_iterations:
                print(f"Max iterations ({self.max_iterations}) reached. W_∞ = {w_inf:.2e}")
            else:
                print(f"Converged in {iters} iterations. W_∞ = {w_inf:.2e}")

        return final_mu
