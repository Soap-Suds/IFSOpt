#!/usr/bin/env python3
"""
Test script to verify correctness of the optimized solver.

This script generates the Sierpinski triangle and validates the output.
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import jax
import jax.numpy as jnp
import numpy as np
from fixed_measure import FixedMeasureSolver
from ifs_solver.utils import create_sierpinski_ifs, visualize_measure


def test_sierpinski_triangle(d: int = 512, visualize: bool = True):
    """Test Sierpinski triangle generation."""
    print("=" * 70)
    print("CORRECTNESS TEST: Sierpinski Triangle")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend().upper()}")
    print(f"Grid resolution: {d}x{d}")
    print()

    # Create solver
    solver = FixedMeasureSolver(d=d, eps=1e-6, max_iterations=300, min_iterations=50)

    # Get Sierpinski IFS
    F, p = create_sierpinski_ifs()

    # Solve
    print("Solving for fixed measure...")
    mu_fixed = solver.solve(F=F, p=p, verbose=True)
    print()

    # Validate properties
    print("Validating solution properties:")
    print(f"  - Total mass: {mu_fixed.sum():.6f} (should be ~1.0)")
    print(f"  - Min value: {mu_fixed.min():.6e}")
    print(f"  - Max value: {mu_fixed.max():.6e}")
    print(f"  - Shape: {mu_fixed.shape}")

    # Check mass conservation
    mass_error = abs(mu_fixed.sum() - 1.0)
    if mass_error < 1e-6:
        print(f"  ✓ Mass conservation: PASS (error: {mass_error:.2e})")
    else:
        print(f"  ✗ Mass conservation: FAIL (error: {mass_error:.2e})")

    # Check characteristic Sierpinski structure
    # The Sierpinski triangle should have zero mass in the center
    center_idx = d // 2
    quarter_size = d // 4
    center_region = mu_fixed[
        center_idx - quarter_size:center_idx + quarter_size,
        center_idx - quarter_size:center_idx + quarter_size
    ]
    center_mass = center_region.sum()
    total_mass = mu_fixed.sum()

    print(f"  - Center region mass ratio: {center_mass / total_mass:.4f}")
    if center_mass / total_mass < 0.05:  # Center should be mostly empty
        print(f"  ✓ Sierpinski structure: PASS (center is sparse)")
    else:
        print(f"  ✗ Sierpinski structure: WARNING (center has unexpected mass)")

    print()

    if visualize:
        print("Generating visualization...")
        visualize_measure(
            mu_fixed,
            title=f"Sierpinski Triangle (d={d})",
            save_path=f"/home/spud/IFSOpt/examples/sierpinski_{d}.png"
        )

    return mu_fixed


def test_multiple_resolutions():
    """Test correctness at multiple resolutions."""
    print("\n" + "=" * 70)
    print("MULTI-RESOLUTION TEST")
    print("=" * 70)

    resolutions = [128, 256, 512]
    F, p = create_sierpinski_ifs()

    for d in resolutions:
        print(f"\nTesting d={d}...")
        solver = FixedMeasureSolver(d=d, eps=1e-6, max_iterations=200, min_iterations=30)
        mu = solver.solve(F=F, p=p, verbose=False)

        mass_error = abs(mu.sum() - 1.0)
        status = "✓ PASS" if mass_error < 1e-6 else "✗ FAIL"
        print(f"  Resolution {d}x{d}: {status} (mass error: {mass_error:.2e})")


if __name__ == "__main__":
    # Test 1: Main Sierpinski test
    mu = test_sierpinski_triangle(d=512, visualize=True)

    # Test 2: Multiple resolutions
    test_multiple_resolutions()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
