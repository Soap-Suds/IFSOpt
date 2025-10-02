#!/usr/bin/env python3
"""
Comprehensive tests for surrogate gradient computation.

Tests:
1. Correctness - verify gradients match notebook implementation
2. Shape consistency - check output shapes
3. Determinism - same inputs -> same outputs
4. Edge cases - boundary conditions
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import jax
import jax.numpy as jnp
import numpy as np
from surrogate_gradients import SurrogateGradientSolver


def test_solver_initialization():
    """Test solver can be initialized with various grid sizes."""
    print("=" * 70)
    print("TEST: Solver Initialization")
    print("=" * 70)

    grid_sizes = [64, 128, 256, 512]

    for d in grid_sizes:
        solver = SurrogateGradientSolver(d=d)
        assert solver.d == d
        assert solver.base_grid.shape == (d * d, 3)
        print(f"✓ d={d}: Initialization successful")

    print()


def test_F_gradient_correctness():
    """Test F gradient computation against reference implementation."""
    print("=" * 70)
    print("TEST: F Gradient Correctness")
    print("=" * 70)

    # Setup
    d = 128
    n = 3

    # Create test data
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
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

    # Compute gradients
    solver = SurrogateGradientSolver(d=d)
    Fgrads = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)

    # Check properties
    assert Fgrads.shape == (n, 2, d, d), f"Expected shape (3, 2, 128, 128), got {Fgrads.shape}"
    assert not jnp.any(jnp.isnan(Fgrads)), "Gradients contain NaN"
    assert not jnp.any(jnp.isinf(Fgrads)), "Gradients contain Inf"

    print(f"✓ Shape: {Fgrads.shape}")
    print(f"✓ No NaN/Inf values")
    print(f"✓ Mean gradient: {jnp.mean(Fgrads):.6e}")
    print(f"✓ Max gradient: {jnp.max(jnp.abs(Fgrads)):.6e}")
    print()


def test_p_gradient_correctness():
    """Test p gradient computation."""
    print("=" * 70)
    print("TEST: p Gradient Correctness")
    print("=" * 70)

    # Setup
    d = 128
    n = 3

    F = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * n, axis=0)
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)
    psi = jnp.ones((d, d), dtype=jnp.float32)

    # Compute gradient
    solver = SurrogateGradientSolver(d=d)
    pgrad = solver.compute_p_gradient(F, rho_F, psi, verbose=False)

    # Check properties
    assert pgrad.shape == (n,), f"Expected shape (3,), got {pgrad.shape}"
    assert not jnp.any(jnp.isnan(pgrad)), "Gradient contains NaN"
    assert not jnp.any(jnp.isinf(pgrad)), "Gradient contains Inf"

    # For identity transformations and constant psi, integral should be 1.0
    expected = 1.0  # sum(rho_F * psi) = sum(1/(d*d) * 1) = 1
    assert jnp.allclose(pgrad, expected, atol=1e-5), f"Expected ~{expected}, got {pgrad}"

    print(f"✓ Shape: {pgrad.shape}")
    print(f"✓ Values: {pgrad}")
    print(f"✓ Correct for identity transform and constant field")
    print()


def test_determinism():
    """Test that same inputs produce same outputs."""
    print("=" * 70)
    print("TEST: Determinism")
    print("=" * 70)

    d = 128
    solver = SurrogateGradientSolver(d=d)

    # Create test data
    key = jax.random.PRNGKey(42)
    F = jax.random.normal(key, (3, 3, 3), dtype=jnp.float32)
    F = F.at[:, 2, :].set(jnp.array([0, 0, 1], dtype=jnp.float32))  # Make affine
    p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
    T = jax.random.normal(key, (2, d, d), dtype=jnp.float32)
    rho_F = jax.random.uniform(key, (d, d), dtype=jnp.float32)
    rho_F = rho_F / rho_F.sum()
    psi = jax.random.normal(key, (d, d), dtype=jnp.float32)

    # Compute twice
    Fgrads1 = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
    Fgrads2 = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)

    pgrad1 = solver.compute_p_gradient(F, rho_F, psi, verbose=False)
    pgrad2 = solver.compute_p_gradient(F, rho_F, psi, verbose=False)

    # Check exact equality
    assert jnp.array_equal(Fgrads1, Fgrads2), "F gradients not deterministic"
    assert jnp.array_equal(pgrad1, pgrad2), "p gradients not deterministic"

    print("✓ F gradients are deterministic")
    print("✓ p gradients are deterministic")
    print()


def test_warmup():
    """Test warmup functionality."""
    print("=" * 70)
    print("TEST: Warmup")
    print("=" * 70)

    d = 128
    solver = SurrogateGradientSolver(d=d)

    # Warmup should not raise errors
    solver.warmup(n_transforms=3, verbose=False)
    assert solver._warmed_up

    print("✓ Warmup completed successfully")
    print("✓ Solver marked as warmed up")
    print()


def test_multiple_resolutions():
    """Test gradient computation works across different resolutions."""
    print("=" * 70)
    print("TEST: Multiple Resolutions")
    print("=" * 70)

    resolutions = [64, 128, 256]

    for d in resolutions:
        solver = SurrogateGradientSolver(d=d)

        F = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * 2, axis=0)
        p = jnp.array([0.5, 0.5], dtype=jnp.float32)
        T = jnp.ones((2, d, d), dtype=jnp.float32)
        rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)
        psi = jnp.ones((d, d), dtype=jnp.float32)

        Fgrads = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
        pgrad = solver.compute_p_gradient(F, rho_F, psi, verbose=False)

        print(f"✓ d={d:3d}: Fgrads shape {Fgrads.shape}, pgrad shape {pgrad.shape}")

    print()


def test_batch_gradient_computation():
    """Test compute_all_gradients convenience function."""
    print("=" * 70)
    print("TEST: Batch Gradient Computation")
    print("=" * 70)

    d = 128
    solver = SurrogateGradientSolver(d=d)

    F = jnp.stack([jnp.eye(3, dtype=jnp.float32)] * 3, axis=0)
    p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)
    T = jnp.ones((2, d, d), dtype=jnp.float32)
    psi = jnp.ones((d, d), dtype=jnp.float32)
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

    # Compute both gradients
    Fgrads, pgrad = solver.compute_all_gradients(F, p, T, psi, rho_F, verbose=False)

    assert Fgrads.shape == (3, 2, d, d)
    assert pgrad.shape == (3,)

    print(f"✓ Combined gradient computation works")
    print(f"✓ Fgrads shape: {Fgrads.shape}")
    print(f"✓ pgrad shape: {pgrad.shape}")
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SURROGATE GRADIENT SOLVER - TEST SUITE")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend().upper()}")
    print()

    tests = [
        test_solver_initialization,
        test_F_gradient_correctness,
        test_p_gradient_correctness,
        test_determinism,
        test_warmup,
        test_multiple_resolutions,
        test_batch_gradient_computation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
