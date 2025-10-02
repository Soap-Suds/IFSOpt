"""
Core implementation of surrogate gradient computation.

Design Philosophy:
-----------------
1. **Static/Dynamic Separation**: Grid size (d) is static, IFS parameters (F, p) are dynamic
2. **Base Grid Caching**: Grid is pre-computed once and reused across all gradient calls
3. **Vectorized Operations**: Use vmap for parallelization across transformations
4. **JIT Compilation**: All hot-path functions are JIT-compiled with appropriate static args
5. **Memory Efficiency**: Minimize data movement between CPU and GPU

Performance Optimizations:
-------------------------
- Base grid computed once at initialization (not per call)
- Coordinate transformation parallelized via vmap
- Single JIT boundary for entire gradient computation
- Explicit static_argnames to prevent unnecessary recompilation
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


# ============================================================================
# Static JIT-compiled functions (core computational kernels)
# ============================================================================

@partial(jax.jit, static_argnames=['d'])
def _create_base_grid(d: int) -> jnp.ndarray:
    """
    Create base grid in homogeneous coordinates.

    This is called once at initialization and cached.

    Args:
        d: Grid dimension

    Returns:
        Base grid (d*d, 3) in homogeneous coordinates
    """
    y_coords = jnp.linspace(0.0, 1.0, d, dtype=jnp.float32)
    x_coords = jnp.linspace(0.0, 1.0, d, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(y_coords, x_coords, indexing='ij')

    base_grid = jnp.stack([
        grid_x.ravel(),
        grid_y.ravel(),
        jnp.ones(d * d, dtype=jnp.float32)
    ], axis=1)

    return base_grid


@partial(jax.jit, static_argnames=['d'])
def _compute_transformed_coords(f: jnp.ndarray, base_grid: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Compute transformed coordinates for a single transformation.

    Args:
        f: Transformation matrix (3, 3)
        base_grid: Base grid in homogeneous coordinates (d*d, 3)
        d: Grid dimension

    Returns:
        Pixel coordinates (2, d, d) for map_coordinates
    """
    # Apply transformation: (3, 3) @ (3, d*d) = (3, d*d)
    transformed = f @ base_grid.T

    # De-homogenize
    x_norm = transformed[0] / transformed[2]
    y_norm = transformed[1] / transformed[2]

    # Convert to pixel indices [0, d-1]
    x_pixel = x_norm * (d - 1.0)
    y_pixel = y_norm * (d - 1.0)

    # Stack as (y, x) for map_coordinates (row-major indexing)
    coords = jnp.stack([
        y_pixel.reshape(d, d),
        x_pixel.reshape(d, d)
    ], axis=0)

    return coords


@partial(jax.jit, static_argnames=['d'])
def _pullback_vector_field_single(T: jnp.ndarray, f: jnp.ndarray,
                                   base_grid: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Pull back vector field T through single transformation f.

    Args:
        T: Vector field (2, d, d)
        f: Transformation matrix (3, 3)
        base_grid: Base grid (d*d, 3)
        d: Grid dimension

    Returns:
        Pulled-back vector field (2, d, d)
    """
    coords = _compute_transformed_coords(f, base_grid, d)

    # Pull back both components using bilinear interpolation
    T_y_pulled = jax.scipy.ndimage.map_coordinates(
        T[0], coords, order=1, mode='constant', cval=0.0
    )
    T_x_pulled = jax.scipy.ndimage.map_coordinates(
        T[1], coords, order=1, mode='constant', cval=0.0
    )

    return jnp.stack([T_y_pulled, T_x_pulled], axis=0)


@partial(jax.jit, static_argnames=['d'])
def _pullback_scalar_field_single(psi: jnp.ndarray, f: jnp.ndarray,
                                   base_grid: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Pull back scalar field psi through single transformation f.

    Args:
        psi: Scalar field (d, d)
        f: Transformation matrix (3, 3)
        base_grid: Base grid (d*d, 3)
        d: Grid dimension

    Returns:
        Pulled-back scalar field (d, d)
    """
    coords = _compute_transformed_coords(f, base_grid, d)

    psi_pulled = jax.scipy.ndimage.map_coordinates(
        psi, coords, order=1, mode='constant', cval=0.0
    )

    return psi_pulled


@partial(jax.jit, static_argnames=['d'])
def _compute_F_gradients_kernel(F: jnp.ndarray, p: jnp.ndarray, T: jnp.ndarray,
                                rho_F: jnp.ndarray, base_grid: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Core kernel for computing F gradients.

    Fully JIT-compiled, vectorized across all transformations.

    Args:
        F: Transformation matrices (n, 3, 3)
        p: Probability vector (n,)
        T: Gradient vector field (2, d, d)
        rho_F: Fixed measure (d, d)
        base_grid: Pre-computed base grid (d*d, 3)
        d: Grid dimension

    Returns:
        F gradients (n, 2, d, d)
    """
    # Vectorized pullback across all transformations
    T_pulled_all = jax.vmap(
        lambda f: _pullback_vector_field_single(T, f, base_grid, d)
    )(F)  # (n, 2, d, d)

    # Compute gradients: p_i * rho_F * T_pulled_i
    # Broadcasting: (n, 1, 1, 1) * (1, 1, d, d) * (n, 2, d, d)
    Fgrads = p[:, None, None, None] * rho_F[None, None, :, :] * T_pulled_all

    return Fgrads


@partial(jax.jit, static_argnames=['d'])
def _compute_p_gradient_kernel(F: jnp.ndarray, rho_F: jnp.ndarray, psi: jnp.ndarray,
                               base_grid: jnp.ndarray, d: int) -> jnp.ndarray:
    """
    Core kernel for computing p gradient.

    Fully JIT-compiled, vectorized across all transformations.

    Args:
        F: Transformation matrices (n, 3, 3)
        rho_F: Fixed measure (d, d)
        psi: Auxiliary potential (d, d)
        base_grid: Pre-computed base grid (d*d, 3)
        d: Grid dimension

    Returns:
        p gradient (n,)
    """
    # Vectorized pullback of psi across all transformations
    psi_pulled_all = jax.vmap(
        lambda f: _pullback_scalar_field_single(psi, f, base_grid, d)
    )(F)  # (n, d, d)

    # Compute integral: sum over grid of rho_F(x) * psi(f_i(x))
    integrand = rho_F[None, :, :] * psi_pulled_all  # (n, d, d)
    pgrad = jnp.sum(integrand, axis=(1, 2))  # (n,)

    return pgrad


# ============================================================================
# Solver Class (manages state and provides clean API)
# ============================================================================

class SurrogateGradientSolver:
    """
    High-performance surrogate gradient solver for IFS optimization.

    This solver is designed for use in optimization loops where F and p change
    frequently. The solver pre-computes the base grid once and reuses it across
    all gradient computations, avoiding redundant work.

    Design Principles:
    -----------------
    1. **Initialization Once**: Create solver with grid size, use many times
    2. **Static Base Grid**: Grid computed once at init, cached for reuse
    3. **JIT Compiled Kernels**: All computational hot-paths are JIT-compiled
    4. **Vectorized Operations**: Parallel processing across transformations
    5. **Minimal Recompilation**: Only d is static; F, p, T, psi are dynamic

    Usage:
    ------
    ```python
    # Initialize once
    solver = SurrogateGradientSolver(d=512)

    # Optionally warmup (triggers JIT compilation)
    solver.warmup(n_transforms=3)

    # Use in optimization loop (fast!)
    for step in range(optimization_steps):
        # Compute fixed measure
        rho_F = fixed_measure_solver.solve(F=F, p=p)

        # Get OT potentials
        T = compute_gradient_field(brenier_potential, d)
        psi = auxiliary_potential.reshape(d, d)

        # Compute gradients (no recompilation!)
        Fgrads = solver.compute_F_gradients(F, p, T, rho_F)
        pgrad = solver.compute_p_gradient(F, rho_F, psi)

        # Update parameters
        F = F - lr * process_F_gradients(Fgrads)
        p = p - lr * pgrad
    ```
    """

    def __init__(self, d: int):
        """
        Initialize the surrogate gradient solver.

        Args:
            d: Grid dimension (must be power of 2 for efficiency)
        """
        if not (d & (d - 1) == 0 and d != 0):
            raise ValueError(f"Grid dimension d={d} should be a power of 2 for optimal performance")

        self.d = d

        # Pre-compute and cache the base grid
        # This is the key optimization: grid is created once, not per call
        self.base_grid = _create_base_grid(d)

        self._warmed_up = False

    def warmup(self, n_transforms: int = 3, verbose: bool = True) -> None:
        """
        Warm up the solver by triggering JIT compilation with dummy data.

        This is optional but recommended before optimization loops to avoid
        compilation overhead on the first gradient computation.

        Args:
            n_transforms: Number of transformations in your IFS
            verbose: Print warmup status
        """
        if verbose:
            print(f"Warming up SurrogateGradientSolver (d={self.d}, n={n_transforms})...")

        # Create dummy data
        dummy_F = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * n_transforms, axis=0)
        dummy_p = jnp.ones(n_transforms, dtype=jnp.float32) / n_transforms
        dummy_T = jnp.zeros((2, self.d, self.d), dtype=jnp.float32)
        dummy_psi = jnp.zeros((self.d, self.d), dtype=jnp.float32)
        dummy_rho_F = jnp.ones((self.d, self.d), dtype=jnp.float32) / (self.d ** 2)

        # Trigger compilation (results discarded)
        _ = self.compute_F_gradients(dummy_F, dummy_p, dummy_T, dummy_rho_F, verbose=False)
        _ = self.compute_p_gradient(dummy_F, dummy_rho_F, dummy_psi, verbose=False)

        self._warmed_up = True
        if verbose:
            print(f"✓ Solver warmed up! Subsequent calls will be fast.")

    def compute_F_gradients(self, F: jnp.ndarray, p: jnp.ndarray, T: jnp.ndarray,
                           rho_F: jnp.ndarray, verbose: bool = True) -> jnp.ndarray:
        """
        Compute gradients with respect to transformation matrices F.

        Formula: grad_i = p_i * rho_F(x) * T(f_i(x))

        Args:
            F: Transformation matrices (n, 3, 3)
            p: Probability vector (n,)
            T: Gradient vector field (2, d, d) from Brenier potential
            rho_F: Fixed measure (d, d)
            verbose: Print computation info

        Returns:
            Fgrads: Gradient vector fields (n, 2, d, d)
        """
        Fgrads = _compute_F_gradients_kernel(F, p, T, rho_F, self.base_grid, self.d)

        # Wait for GPU computation to complete
        Fgrads.block_until_ready()

        if verbose and not self._warmed_up:
            print("Note: First call includes JIT compilation overhead. Use warmup() to avoid this.")

        return Fgrads

    def compute_p_gradient(self, F: jnp.ndarray, rho_F: jnp.ndarray, psi: jnp.ndarray,
                          verbose: bool = True) -> jnp.ndarray:
        """
        Compute gradient with respect to probability vector p.

        Formula: grad_p[i] = ∫_X rho_F(x) * psi(f_i(x)) dx

        Args:
            F: Transformation matrices (n, 3, 3)
            rho_F: Fixed measure (d, d)
            psi: Auxiliary potential (d, d) - scalar field
            verbose: Print computation info

        Returns:
            pgrad: Gradient scalars (n,)
        """
        pgrad = _compute_p_gradient_kernel(F, rho_F, psi, self.base_grid, self.d)

        # Wait for GPU computation to complete
        pgrad.block_until_ready()

        if verbose and not self._warmed_up:
            print("Note: First call includes JIT compilation overhead. Use warmup() to avoid this.")

        return pgrad

    def compute_all_gradients(self, F: jnp.ndarray, p: jnp.ndarray,
                             T: jnp.ndarray, psi: jnp.ndarray, rho_F: jnp.ndarray,
                             verbose: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute both F and p gradients in one call.

        Note: This doesn't save computation (they are independent), but provides
        a convenient single-call interface.

        Args:
            F: Transformation matrices (n, 3, 3)
            p: Probability vector (n,)
            T: Gradient vector field (2, d, d)
            psi: Auxiliary potential (d, d)
            rho_F: Fixed measure (d, d)
            verbose: Print computation info

        Returns:
            Fgrads: Gradient vector fields (n, 2, d, d)
            pgrad: Gradient scalars (n,)
        """
        Fgrads = self.compute_F_gradients(F, p, T, rho_F, verbose=False)
        pgrad = self.compute_p_gradient(F, rho_F, psi, verbose=False)

        if verbose and not self._warmed_up:
            print("Note: First call includes JIT compilation overhead. Use warmup() to avoid this.")

        return Fgrads, pgrad
