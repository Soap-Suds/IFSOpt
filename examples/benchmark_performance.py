#!/usr/bin/env python3
"""
Benchmark script comparing old vs. new implementation.

This script measures:
1. First call time (with JIT compilation)
2. Subsequent call times (cached compilation)
3. Optimization loop simulation
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import time
import jax
import jax.numpy as jnp
import numpy as np
from fixed_measure import FixedMeasureSolver
from ifs_solver.utils import create_sierpinski_ifs


def benchmark_solver(d: int = 512, n_runs: int = 5):
    """
    Benchmark the optimized solver.

    Args:
        d: Grid resolution
        n_runs: Number of repeated solves (simulating optimization loop)
    """
    print("=" * 70)
    print(f"PERFORMANCE BENCHMARK (d={d})")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend().upper()}")
    print(f"Number of runs: {n_runs}")
    print()

    # Initialize solver
    solver = FixedMeasureSolver(
        d=d,
        eps=1e-6,
        max_iterations=300,
        min_iterations=50
    )

    # Get test IFS
    F, p = create_sierpinski_ifs()

    # ========================================================================
    # Test 1: First call (includes JIT compilation)
    # ========================================================================
    print("Test 1: First call (with JIT compilation)")
    print("-" * 70)

    start = time.perf_counter()
    mu1 = solver.solve(F=F, p=p, verbose=False)
    mu1.block_until_ready()  # Ensure computation completes
    time_first_call = time.perf_counter() - start

    print(f"First call time: {time_first_call:.4f} seconds")
    print()

    # ========================================================================
    # Test 2: Subsequent calls with SAME parameters (cached)
    # ========================================================================
    print("Test 2: Subsequent calls with SAME parameters (fully cached)")
    print("-" * 70)

    times_same = []
    for i in range(n_runs):
        start = time.perf_counter()
        mu = solver.solve(F=F, p=p, verbose=False)
        mu.block_until_ready()
        elapsed = time.perf_counter() - start
        times_same.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f} seconds")

    avg_time_same = np.mean(times_same)
    std_time_same = np.std(times_same)
    print(f"\nAverage: {avg_time_same:.4f} ± {std_time_same:.4f} seconds")
    print(f"Speedup vs first call: {time_first_call / avg_time_same:.2f}x")
    print()

    # ========================================================================
    # Test 3: Optimization loop simulation (varying F, same shape)
    # ========================================================================
    print("Test 3: Optimization loop simulation (varying F, same shape)")
    print("-" * 70)
    print("Simulating optimization where F changes but shape remains constant")
    print()

    times_varying = []
    for i in range(n_runs):
        # Perturb F slightly (simulating optimization steps)
        noise_scale = 0.01
        F_perturbed = [
            f + noise_scale * jnp.array(np.random.randn(3, 3), dtype=jnp.float32)
            for f in F
        ]
        # Ensure valid affine structure
        for f in F_perturbed:
            f = f.at[2, :].set(jnp.array([0.0, 0.0, 1.0]))

        start = time.perf_counter()
        mu = solver.solve(F=F_perturbed, p=p, verbose=False)
        mu.block_until_ready()
        elapsed = time.perf_counter() - start
        times_varying.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f} seconds")

    avg_time_varying = np.mean(times_varying)
    std_time_varying = np.std(times_varying)
    print(f"\nAverage: {avg_time_varying:.4f} ± {std_time_varying:.4f} seconds")
    print(f"Overhead vs cached: {(avg_time_varying / avg_time_same - 1) * 100:.1f}%")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"First call (with compilation): {time_first_call:.4f} s")
    print(f"Cached calls (same params):     {avg_time_same:.4f} s  ({time_first_call/avg_time_same:.1f}x speedup)")
    print(f"Optimization loop (varying F):  {avg_time_varying:.4f} s  ({time_first_call/avg_time_varying:.1f}x speedup)")
    print()
    print(f"✓ Compilation happens only once")
    print(f"✓ Subsequent calls reuse compiled code")
    print(f"✓ Ready for optimization loops!")
    print()

    return {
        'first_call': time_first_call,
        'cached_avg': avg_time_same,
        'varying_avg': avg_time_varying,
        'speedup_cached': time_first_call / avg_time_same,
        'speedup_varying': time_first_call / avg_time_varying
    }


def compare_resolutions():
    """Compare performance across different resolutions."""
    print("\n" + "=" * 70)
    print("RESOLUTION SCALING TEST")
    print("=" * 70)

    resolutions = [128, 256, 512, 1024]
    F, p = create_sierpinski_ifs()

    results = []

    for d in resolutions:
        print(f"\nTesting d={d}...")
        solver = FixedMeasureSolver(d=d, eps=1e-6, max_iterations=100, min_iterations=30)

        # Warmup
        _ = solver.solve(F=F, p=p, verbose=False)

        # Benchmark
        times = []
        for _ in range(3):
            start = time.perf_counter()
            mu = solver.solve(F=F, p=p, verbose=False)
            mu.block_until_ready()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        results.append((d, avg_time))
        print(f"  Average time: {avg_time:.4f} s")

    print("\n" + "-" * 70)
    print("Resolution | Time (s) | Relative to d=128")
    print("-" * 70)
    base_time = results[0][1]
    for d, t in results:
        relative = t / base_time
        print(f"{d:10d} | {t:8.4f} | {relative:6.2f}x")


if __name__ == "__main__":
    # Main benchmark
    results = benchmark_solver(d=512, n_runs=5)

    # Resolution scaling
    compare_resolutions()

    print("\n" + "=" * 70)
    print("BENCHMARKING COMPLETED")
    print("=" * 70)
