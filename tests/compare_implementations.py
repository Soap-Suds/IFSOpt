#!/usr/bin/env python3
"""
Compare old (notebook) vs new (optimized) implementation.

This script verifies:
1. Correctness: outputs match
2. Performance: new version is faster
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import time
import jax
import jax.numpy as jnp
import numpy as np
import functools
from surrogate_gradients import SurrogateGradientSolver


# ============================================================================
# OLD IMPLEMENTATION (from notebook)
# ============================================================================

@functools.partial(jax.jit, static_argnames=['d'])
def _compute_transformed_coords_old(f, base_grid, d):
    transformed = f @ base_grid.T
    x_norm = transformed[0] / transformed[2]
    y_norm = transformed[1] / transformed[2]
    x_pixel = x_norm * (d - 1.0)
    y_pixel = y_norm * (d - 1.0)
    coords = jnp.stack([
        y_pixel.reshape(d, d),
        x_pixel.reshape(d, d)
    ], axis=0)
    return coords


@functools.partial(jax.jit, static_argnames=['d'])
def _pullback_vector_field_old(T, f, base_grid, d):
    coords = _compute_transformed_coords_old(f, base_grid, d)
    T_y_pulled = jax.scipy.ndimage.map_coordinates(
        T[0], coords, order=1, mode='constant', cval=0.0
    )
    T_x_pulled = jax.scipy.ndimage.map_coordinates(
        T[1], coords, order=1, mode='constant', cval=0.0
    )
    return jnp.stack([T_y_pulled, T_x_pulled], axis=0)


@functools.partial(jax.jit, static_argnames=['d'])
def _pullback_scalar_field_old(psi, f, base_grid, d):
    coords = _compute_transformed_coords_old(f, base_grid, d)
    psi_pulled = jax.scipy.ndimage.map_coordinates(
        psi, coords, order=1, mode='constant', cval=0.0
    )
    return psi_pulled


@functools.partial(jax.jit, static_argnames=['d'])
def IFSgradient_F_only_old(F, p, T, rho_F, d):
    """Old implementation (creates base grid every call)."""
    # Create base grid every time (inefficient!)
    y_coords = jnp.linspace(0.0, 1.0, d, dtype=jnp.float32)
    x_coords = jnp.linspace(0.0, 1.0, d, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(y_coords, x_coords, indexing='ij')

    base_grid = jnp.stack([
        grid_x.ravel(),
        grid_y.ravel(),
        jnp.ones(d * d, dtype=jnp.float32)
    ], axis=1)

    T_pulled_all = jax.vmap(
        lambda f: _pullback_vector_field_old(T, f, base_grid, d)
    )(F)

    Fgrads = p[:, None, None, None] * rho_F[None, None, :, :] * T_pulled_all
    return Fgrads


@functools.partial(jax.jit, static_argnames=['d'])
def compute_p_gradient_old(F, rho_F, psi, d):
    """Old implementation (creates base grid every call)."""
    # Create base grid every time (inefficient!)
    y_coords = jnp.linspace(0.0, 1.0, d, dtype=jnp.float32)
    x_coords = jnp.linspace(0.0, 1.0, d, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(y_coords, x_coords, indexing='ij')

    base_grid = jnp.stack([
        grid_x.ravel(),
        grid_y.ravel(),
        jnp.ones(d * d, dtype=jnp.float32)
    ], axis=1)

    psi_pulled_all = jax.vmap(
        lambda f: _pullback_scalar_field_old(psi, f, base_grid, d)
    )(F)

    integrand = rho_F[None, :, :] * psi_pulled_all
    pgrad = jnp.sum(integrand, axis=(1, 2))
    return pgrad


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_correctness(d=256):
    """Compare outputs of old vs new implementation."""
    print("=" * 70)
    print("CORRECTNESS COMPARISON")
    print("=" * 70)
    print(f"Resolution: {d}x{d}\n")

    # Test data
    F = jnp.stack([
        jnp.array([[0.5, 0.0, 0.0],
                   [0.0, 0.5, 0.0],
                   [0.0, 0.0, 1.0]], dtype=jnp.float32),
        jnp.array([[0.5, 0.0, 0.5],
                   [0.0, 0.5, 0.0],
                   [0.0, 0.0, 1.0]], dtype=jnp.float32),
        jnp.array([[0.5, 0.0, 0.0],
                   [0.0, 0.5, 0.5],
                   [0.0, 0.0, 1.0]], dtype=jnp.float32)
    ], axis=0)

    p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
    T = jnp.ones((2, d, d), dtype=jnp.float32) * 0.1
    psi = jnp.ones((d, d), dtype=jnp.float32) * 0.1
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

    # Old implementation
    print("Running OLD implementation...")
    Fgrads_old = IFSgradient_F_only_old(F, p, T, rho_F, d=d)
    pgrad_old = compute_p_gradient_old(F, rho_F, psi, d=d)
    Fgrads_old.block_until_ready()
    pgrad_old.block_until_ready()

    # New implementation
    print("Running NEW implementation...")
    solver = SurrogateGradientSolver(d=d)
    Fgrads_new = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
    pgrad_new = solver.compute_p_gradient(F, rho_F, psi, verbose=False)
    Fgrads_new.block_until_ready()
    pgrad_new.block_until_ready()

    # Compare
    print("\nComparing outputs...")
    Fgrads_diff = jnp.max(jnp.abs(Fgrads_old - Fgrads_new))
    pgrad_diff = jnp.max(jnp.abs(pgrad_old - pgrad_new))

    print(f"  F gradients max difference: {Fgrads_diff:.2e}")
    print(f"  p gradient max difference:  {pgrad_diff:.2e}")

    if Fgrads_diff < 1e-6 and pgrad_diff < 1e-6:
        print("  ✓ PASS: Outputs match (diff < 1e-6)")
    else:
        print("  ✗ FAIL: Outputs differ significantly")

    print()


def compare_performance(d=256, n_iterations=10):
    """Compare performance in optimization loop scenario."""
    print("=" * 70)
    print("PERFORMANCE COMPARISON (Optimization Loop Simulation)")
    print("=" * 70)
    print(f"Resolution: {d}x{d}")
    print(f"Iterations: {n_iterations}\n")

    # Test data
    F = jnp.stack([
        jnp.array([[0.5, 0.0, 0.0],
                   [0.0, 0.5, 0.0],
                   [0.0, 0.0, 1.0]], dtype=jnp.float32),
        jnp.array([[0.5, 0.0, 0.5],
                   [0.0, 0.5, 0.0],
                   [0.0, 0.0, 1.0]], dtype=jnp.float32),
        jnp.array([[0.5, 0.0, 0.0],
                   [0.0, 0.5, 0.5],
                   [0.0, 0.0, 1.0]], dtype=jnp.float32)
    ], axis=0)

    p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
    T = jnp.ones((2, d, d), dtype=jnp.float32) * 0.1
    psi = jnp.ones((d, d), dtype=jnp.float32) * 0.1
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

    # ========================================================================
    # OLD IMPLEMENTATION
    # ========================================================================
    print("OLD Implementation (recreates base grid each call)")
    print("-" * 70)

    times_old = []
    for i in range(n_iterations):
        start = time.perf_counter()
        Fgrads = IFSgradient_F_only_old(F, p, T, rho_F, d=d)
        pgrad = compute_p_gradient_old(F, rho_F, psi, d=d)
        Fgrads.block_until_ready()
        pgrad.block_until_ready()
        elapsed = time.perf_counter() - start
        times_old.append(elapsed)
        if i < 3:
            print(f"  Iteration {i+1}: {elapsed:.4f}s")

    avg_old = np.mean(times_old)
    print(f"Average: {avg_old:.4f}s (±{np.std(times_old):.4f}s)\n")

    # ========================================================================
    # NEW IMPLEMENTATION
    # ========================================================================
    print("NEW Implementation (cached base grid)")
    print("-" * 70)

    solver = SurrogateGradientSolver(d=d)
    solver.warmup(n_transforms=3, verbose=False)

    times_new = []
    for i in range(n_iterations):
        start = time.perf_counter()
        Fgrads = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
        pgrad = solver.compute_p_gradient(F, rho_F, psi, verbose=False)
        Fgrads.block_until_ready()
        pgrad.block_until_ready()
        elapsed = time.perf_counter() - start
        times_new.append(elapsed)
        if i < 3:
            print(f"  Iteration {i+1}: {elapsed:.4f}s")

    avg_new = np.mean(times_new)
    print(f"Average: {avg_new:.4f}s (±{np.std(times_new):.4f}s)\n")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    speedup = avg_old / avg_new
    improvement = (1 - avg_new / avg_old) * 100

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"OLD implementation: {avg_old:.4f}s")
    print(f"NEW implementation: {avg_new:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}% faster\n")

    if speedup > 1.05:
        print(f"✓ NEW implementation is significantly faster!")
    elif speedup > 0.95:
        print(f"≈ Performance is similar (within 5%)")
    else:
        print(f"Note: Performance may vary based on compilation/caching")

    print()


def run_comparison():
    """Run all comparisons."""
    print("\n" + "=" * 70)
    print("SURROGATE GRADIENTS: OLD vs NEW COMPARISON")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend().upper()}\n")

    compare_correctness(d=256)
    compare_performance(d=256, n_iterations=10)

    print("=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
