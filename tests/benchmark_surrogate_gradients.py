#!/usr/bin/env python3
"""
Performance benchmarks for surrogate gradient computation.

Benchmarks:
1. First call vs cached (JIT compilation overhead)
2. Different grid sizes
3. Different numbers of transformations
4. Old vs new implementation
5. Optimization loop simulation
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import time
import jax
import jax.numpy as jnp
import numpy as np
from surrogate_gradients import SurrogateGradientSolver


def benchmark_compilation_overhead(d=256, n_runs=5):
    """Measure JIT compilation overhead."""
    print("=" * 70)
    print(f"BENCHMARK: Compilation Overhead (d={d})")
    print("=" * 70)

    solver = SurrogateGradientSolver(d=d)

    # Test data
    F = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * 3, axis=0)
    p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
    T = jnp.ones((2, d, d), dtype=jnp.float32)
    psi = jnp.ones((d, d), dtype=jnp.float32)
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

    # First call (includes compilation)
    print("First call (includes JIT compilation)...")
    start = time.perf_counter()
    Fgrads1 = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
    Fgrads1.block_until_ready()
    time_first_F = time.perf_counter() - start

    start = time.perf_counter()
    pgrad1 = solver.compute_p_gradient(F, rho_F, psi, verbose=False)
    pgrad1.block_until_ready()
    time_first_p = time.perf_counter() - start

    print(f"  F gradients: {time_first_F:.4f}s")
    print(f"  p gradient:  {time_first_p:.4f}s")

    # Cached calls
    print(f"\nCached calls (n={n_runs})...")
    times_F = []
    times_p = []

    for i in range(n_runs):
        start = time.perf_counter()
        Fgrads = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
        Fgrads.block_until_ready()
        times_F.append(time.perf_counter() - start)

        start = time.perf_counter()
        pgrad = solver.compute_p_gradient(F, rho_F, psi, verbose=False)
        pgrad.block_until_ready()
        times_p.append(time.perf_counter() - start)

    avg_F = np.mean(times_F)
    avg_p = np.mean(times_p)

    print(f"  F gradients: {avg_F:.4f}s (±{np.std(times_F):.4f})")
    print(f"  p gradient:  {avg_p:.4f}s (±{np.std(times_p):.4f})")

    print(f"\nSpeedup:")
    print(f"  F gradients: {time_first_F / avg_F:.1f}x")
    print(f"  p gradient:  {time_first_p / avg_p:.1f}x")
    print()


def benchmark_grid_sizes(n_runs=3):
    """Benchmark performance across different grid sizes."""
    print("=" * 70)
    print("BENCHMARK: Grid Size Scaling")
    print("=" * 70)

    grid_sizes = [64, 128, 256, 512]
    results = []

    for d in grid_sizes:
        solver = SurrogateGradientSolver(d=d)
        solver.warmup(n_transforms=3, verbose=False)

        F = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * 3, axis=0)
        p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
        T = jnp.ones((2, d, d), dtype=jnp.float32)
        psi = jnp.ones((d, d), dtype=jnp.float32)
        rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            Fgrads = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
            pgrad = solver.compute_p_gradient(F, rho_F, psi, verbose=False)
            Fgrads.block_until_ready()
            pgrad.block_until_ready()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        results.append((d, avg_time))

        print(f"d={d:4d}: {avg_time:.4f}s (±{np.std(times):.4f}s)")

    # Show scaling
    print(f"\nScaling factor:")
    base_time = results[0][1]
    for d, t in results:
        scale = t / base_time
        expected_scale = (d / results[0][0]) ** 2
        print(f"  d={d:4d}: {scale:5.2f}x  (theoretical: {expected_scale:.2f}x)")

    print()


def benchmark_num_transformations(d=256, n_runs=3):
    """Benchmark performance with different numbers of transformations."""
    print("=" * 70)
    print(f"BENCHMARK: Number of Transformations (d={d})")
    print("=" * 70)

    n_transforms_list = [2, 3, 5, 10]

    for n in n_transforms_list:
        solver = SurrogateGradientSolver(d=d)
        solver.warmup(n_transforms=n, verbose=False)

        F = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * n, axis=0)
        p = jnp.ones(n, dtype=jnp.float32) / n
        T = jnp.ones((2, d, d), dtype=jnp.float32)
        psi = jnp.ones((d, d), dtype=jnp.float32)
        rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            Fgrads = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
            pgrad = solver.compute_p_gradient(F, rho_F, psi, verbose=False)
            Fgrads.block_until_ready()
            pgrad.block_until_ready()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        print(f"n={n:2d} transforms: {avg_time:.4f}s (±{np.std(times):.4f}s)")

    print()


def benchmark_optimization_loop(d=256, n_steps=10):
    """Simulate an optimization loop to measure real-world performance."""
    print("=" * 70)
    print(f"BENCHMARK: Optimization Loop Simulation (d={d}, steps={n_steps})")
    print("=" * 70)

    solver = SurrogateGradientSolver(d=d)
    print("Warming up...")
    solver.warmup(n_transforms=3, verbose=False)

    # Initial IFS
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

    # Dummy OT potentials (in real scenario, these would be recomputed)
    T = jnp.ones((2, d, d), dtype=jnp.float32) * 0.1
    psi = jnp.ones((d, d), dtype=jnp.float32) * 0.1
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

    print("Running optimization loop...")
    times = []

    for step in range(n_steps):
        start = time.perf_counter()

        # Compute gradients
        Fgrads = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
        pgrad = solver.compute_p_gradient(F, rho_F, psi, verbose=False)

        # Wait for completion
        Fgrads.block_until_ready()
        pgrad.block_until_ready()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Simulate parameter update
        # In reality, you'd process the gradients and update F, p
        noise = jax.random.normal(jax.random.PRNGKey(step), F.shape) * 0.01
        F = F + noise
        F = F.at[:, 2, :].set(jnp.array([0, 0, 1], dtype=jnp.float32))

    avg_time = np.mean(times)
    print(f"\nPer-iteration statistics:")
    print(f"  Average: {avg_time:.4f}s")
    print(f"  Std dev: {np.std(times):.4f}s")
    print(f"  Min:     {np.min(times):.4f}s")
    print(f"  Max:     {np.max(times):.4f}s")
    print(f"\nEstimated time for 1000 iterations: {avg_time * 1000:.1f}s ({avg_time * 1000 / 60:.1f} min)")
    print()


def benchmark_warmup_benefit(d=256):
    """Measure benefit of warmup."""
    print("=" * 70)
    print(f"BENCHMARK: Warmup Benefit (d={d})")
    print("=" * 70)

    # Without warmup
    print("Without warmup:")
    solver_no_warmup = SurrogateGradientSolver(d=d)

    F = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * 3, axis=0)
    p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
    T = jnp.ones((2, d, d), dtype=jnp.float32)
    psi = jnp.ones((d, d), dtype=jnp.float32)
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

    start = time.perf_counter()
    Fgrads = solver_no_warmup.compute_F_gradients(F, p, T, rho_F, verbose=False)
    pgrad = solver_no_warmup.compute_p_gradient(F, rho_F, psi, verbose=False)
    Fgrads.block_until_ready()
    pgrad.block_until_ready()
    time_no_warmup = time.perf_counter() - start
    print(f"  First call: {time_no_warmup:.4f}s")

    # With warmup
    print("\nWith warmup:")
    solver_warmup = SurrogateGradientSolver(d=d)

    start = time.perf_counter()
    solver_warmup.warmup(n_transforms=3, verbose=False)
    warmup_time = time.perf_counter() - start
    print(f"  Warmup:     {warmup_time:.4f}s")

    start = time.perf_counter()
    Fgrads = solver_warmup.compute_F_gradients(F, p, T, rho_F, verbose=False)
    pgrad = solver_warmup.compute_p_gradient(F, rho_F, psi, verbose=False)
    Fgrads.block_until_ready()
    pgrad.block_until_ready()
    time_after_warmup = time.perf_counter() - start
    print(f"  First call: {time_after_warmup:.4f}s")

    print(f"\nSpeedup: {time_no_warmup / time_after_warmup:.2f}x")
    print()


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("SURROGATE GRADIENT SOLVER - PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend().upper()}")
    print()

    benchmark_compilation_overhead(d=256, n_runs=5)
    benchmark_grid_sizes(n_runs=3)
    benchmark_num_transformations(d=256, n_runs=3)
    benchmark_warmup_benefit(d=256)
    benchmark_optimization_loop(d=256, n_steps=10)

    print("=" * 70)
    print("BENCHMARKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
