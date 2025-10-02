#!/usr/bin/env python3
"""
Compare old (notebook) vs. new (optimized) implementation.

This script loads both implementations and compares:
1. Correctness (do they produce the same output?)
2. Performance (which is faster?)
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import List, Tuple

# Import new implementation
from fixed_measure import FixedMeasureSolver
from ifs_solver.utils import create_sierpinski_ifs


# ============================================================================
# OLD IMPLEMENTATION (from notebook)
# ============================================================================

class JaxFixedMeasureOld:
    """Original implementation from notebook."""

    def __init__(self, F: List[jnp.ndarray], nu: jnp.ndarray, p: jnp.ndarray = None, eps: float = 1e-4):
        if p is not None and not jnp.allclose(p.sum(), 1.0):
            raise ValueError("Probability vector p must sum to 1.")

        self.n = len(F)
        self.eps = eps

        self.F = jnp.stack(F, axis=0).astype(jnp.float32)

        if p is None:
            self.p = jnp.full((self.n,), 1.0 / self.n, dtype=jnp.float32)
        else:
            assert p.shape[0] == self.n
            self.p = p.astype(jnp.float32)

        assert nu.ndim == 2 and nu.shape[0] == nu.shape[1]
        self.d = nu.shape[0]
        if not (self.d & (self.d - 1) == 0 and self.d != 0):
            raise ValueError(f"d = {self.d} is not a power of 2.")

        self.mu_initial = nu.astype(jnp.float32)
        self.mu_initial = self.mu_initial / self.mu_initial.sum()

        self.F_inv, self.jac_dets, self.base_grid = self._precompute_transforms()

    def _precompute_transforms(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        y_coords = jnp.linspace(0.0, 1.0, self.d, dtype=jnp.float32)
        x_coords = jnp.linspace(0.0, 1.0, self.d, dtype=jnp.float32)
        grid_y, grid_x = jnp.meshgrid(y_coords, x_coords, indexing='ij')

        grid_flat = jnp.stack([
            grid_x.ravel(),
            grid_y.ravel(),
            jnp.ones(self.d * self.d, dtype=jnp.float32)
        ], axis=1)

        base_grid = grid_flat
        F_inv = jnp.linalg.inv(self.F)
        A = self.F[:, :2, :2]
        jac_dets = jnp.abs(jnp.linalg.det(A))

        return F_inv, jac_dets, base_grid

    @partial(jax.jit, static_argnums=(0,))
    def _compute_transformed_grid(self, f_inv: jnp.ndarray) -> jnp.ndarray:
        transformed = f_inv @ self.base_grid.T
        x_norm = transformed[0] / transformed[2]
        y_norm = transformed[1] / transformed[2]
        x_pixel = x_norm * (self.d - 1.0)
        y_pixel = y_norm * (self.d - 1.0)
        coords = jnp.stack([
            y_pixel.reshape(self.d, self.d),
            x_pixel.reshape(self.d, self.d)
        ], axis=0)
        return coords

    @partial(jax.jit, static_argnums=(0,))
    def _single_pushforward(self, mu: jnp.ndarray, f_inv: jnp.ndarray,
                           jac_det: float, prob: float) -> jnp.ndarray:
        coords = self._compute_transformed_grid(f_inv)
        pushed = jax.scipy.ndimage.map_coordinates(
            mu, coords, order=1, mode='constant', cval=0.0
        )
        pushed = jnp.where(jac_det > 1e-9, pushed / jac_det, pushed)
        return prob * pushed

    @partial(jax.jit, static_argnums=(0,))
    def push_forward(self, mu: jnp.ndarray) -> jnp.ndarray:
        pushed_components = jax.vmap(
            lambda f_inv, jac, p: self._single_pushforward(mu, f_inv, jac, p)
        )(self.F_inv, self.jac_dets, self.p)
        mu_next = jnp.sum(pushed_components, axis=0)
        mu_next = mu_next / mu_next.sum()
        return mu_next

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, mu: jnp.ndarray) -> Tuple[jnp.ndarray, float]:
        mu_next = self.push_forward(mu)
        w_inf = jnp.max(jnp.abs(mu - mu_next))
        return mu_next, w_inf

    def solve(self, max_iterations: int = 1000, min_iterations: int = 100) -> jnp.ndarray:
        def cond_fun(state):
            mu, iteration, w_inf = state
            converged = (iteration >= min_iterations) & (w_inf < self.eps)
            not_done = iteration < max_iterations
            return not_done & (~converged)

        def body_fun(state):
            mu, iteration, _ = state
            mu_next, w_inf = self._step(mu)
            return mu_next, iteration + 1, w_inf

        initial_state = (self.mu_initial, 0, jnp.inf)
        final_mu, iters, w_inf = jax.lax.while_loop(cond_fun, body_fun, initial_state)
        final_mu.block_until_ready()
        return final_mu


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_correctness(d: int = 256):
    """Compare outputs of old vs. new implementation."""
    print("=" * 70)
    print("CORRECTNESS COMPARISON")
    print("=" * 70)
    print(f"Resolution: {d}x{d}")
    print()

    # Get test IFS
    F, p = create_sierpinski_ifs()
    nu = jnp.ones((d, d), dtype=jnp.float32)

    # Old implementation
    print("Running OLD implementation...")
    old_solver = JaxFixedMeasureOld(F=F, nu=nu, p=p, eps=1e-6)
    start = time.perf_counter()
    mu_old = old_solver.solve(max_iterations=200, min_iterations=50)
    time_old = time.perf_counter() - start
    print(f"  Time: {time_old:.4f} s")
    print()

    # New implementation
    print("Running NEW implementation...")
    new_solver = FixedMeasureSolver(d=d, eps=1e-6, max_iterations=200, min_iterations=50)
    start = time.perf_counter()
    mu_new = new_solver.solve(F=F, p=p, nu=nu, verbose=False)
    time_new = time.perf_counter() - start
    print(f"  Time: {time_new:.4f} s")
    print()

    # Compare outputs
    print("Comparing outputs...")
    diff = jnp.abs(mu_old - mu_new)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)
    rel_diff = max_diff / jnp.max(mu_old)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    print(f"  Relative difference: {rel_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✓ PASS: Outputs are nearly identical")
    else:
        print("  ✗ WARNING: Outputs differ significantly")

    print()
    return mu_old, mu_new, time_old, time_new


def compare_performance(d: int = 512, n_iterations: int = 5):
    """Compare performance in optimization loop scenario."""
    print("=" * 70)
    print("PERFORMANCE COMPARISON (Optimization Loop Simulation)")
    print("=" * 70)
    print(f"Resolution: {d}x{d}")
    print(f"Iterations: {n_iterations}")
    print()

    F, p = create_sierpinski_ifs()
    nu = jnp.ones((d, d), dtype=jnp.float32)

    # ========================================================================
    # OLD IMPLEMENTATION
    # ========================================================================
    print("OLD Implementation (class-based, instance per call)")
    print("-" * 70)

    times_old = []
    for i in range(n_iterations):
        # In old version, you'd create new instance each time F changes
        start = time.perf_counter()
        solver = JaxFixedMeasureOld(F=F, nu=nu, p=p, eps=1e-6)
        mu = solver.solve(max_iterations=200, min_iterations=50)
        elapsed = time.perf_counter() - start
        times_old.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f} s")

    avg_old = np.mean(times_old)
    std_old = np.std(times_old)
    print(f"Average: {avg_old:.4f} ± {std_old:.4f} s")
    print()

    # ========================================================================
    # NEW IMPLEMENTATION
    # ========================================================================
    print("NEW Implementation (static solver, reusable)")
    print("-" * 70)

    solver_new = FixedMeasureSolver(d=d, eps=1e-6, max_iterations=200, min_iterations=50)

    times_new = []
    for i in range(n_iterations):
        start = time.perf_counter()
        mu = solver_new.solve(F=F, p=p, nu=nu, verbose=False)
        elapsed = time.perf_counter() - start
        times_new.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f} s")

    avg_new = np.mean(times_new)
    std_new = np.std(times_new)
    print(f"Average: {avg_new:.4f} ± {std_new:.4f} s")
    print()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    speedup = avg_old / avg_new
    improvement = (1 - avg_new / avg_old) * 100

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"OLD implementation: {avg_old:.4f} ± {std_old:.4f} s")
    print(f"NEW implementation: {avg_new:.4f} ± {std_new:.4f} s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Improvement: {improvement:.1f}% faster")
    print()

    if speedup > 1.1:
        print(f"✓ NEW implementation is significantly faster!")
    elif speedup > 0.9:
        print(f"≈ Performance is similar (within 10%)")
    else:
        print(f"✗ OLD implementation was faster (investigate!)")

    return avg_old, avg_new, speedup


if __name__ == "__main__":
    # Test 1: Correctness
    mu_old, mu_new, t_old, t_new = compare_correctness(d=256)

    print("\n")

    # Test 2: Performance
    avg_old, avg_new, speedup = compare_performance(d=512, n_iterations=5)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)
