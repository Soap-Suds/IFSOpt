#!/usr/bin/env python3
"""
Complete example showing how to use SurrogateGradientSolver in an optimization loop.

This demonstrates:
1. Solver initialization and warmup
2. Integration with FixedMeasureSolver
3. Computing gradients
4. Basic gradient descent step
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import jax
import jax.numpy as jnp
import numpy as np
from surrogate_gradients import SurrogateGradientSolver


def main():
    print("=" * 70)
    print("SURROGATE GRADIENT SOLVER - COMPLETE EXAMPLE")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend().upper()}\n")

    # ========================================================================
    # Configuration
    # ========================================================================
    d = 256  # Grid dimension
    n_transforms = 3  # Number of IFS transformations

    # ========================================================================
    # Initialize Solver
    # ========================================================================
    print("[1] Initializing SurrogateGradientSolver...")
    solver = SurrogateGradientSolver(d=d)
    print(f"    Grid dimension: {d}x{d}")
    print(f"    Base grid cached: {solver.base_grid.shape}")

    # ========================================================================
    # Warmup (Optional but Recommended)
    # ========================================================================
    print("\n[2] Warming up solver...")
    solver.warmup(n_transforms=n_transforms, verbose=False)
    print("    ✓ Compilation complete")

    # ========================================================================
    # Example IFS (Sierpinski Triangle)
    # ========================================================================
    print("\n[3] Setting up test IFS (Sierpinski triangle)...")
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
    print(f"    Number of transformations: {n_transforms}")
    print(f"    Probabilities: {p}")

    # ========================================================================
    # Simulate Optimization Loop
    # ========================================================================
    print("\n[4] Simulating optimization loop...")
    print("    (In practice, you'd compute these from actual OT solver)")

    # Dummy OT potentials (in practice, from Sinkhorn solver)
    # These would be outputs from your OT solver
    brenier_potential_flat = jnp.ones(d * d, dtype=jnp.float32) * 0.1
    auxiliary_potential_flat = jnp.ones(d * d, dtype=jnp.float32) * 0.1

    # Compute gradient field from Brenier potential
    # (In practice, use your compute_gradient_field function)
    brenier_potential_2d = brenier_potential_flat.reshape(d, d)
    grad_y, grad_x = jnp.gradient(brenier_potential_2d)
    T = jnp.stack([grad_y, grad_x], axis=0)

    # Auxiliary potential
    psi = auxiliary_potential_flat.reshape(d, d)

    # Fixed measure (in practice, from FixedMeasureSolver)
    rho_F = jnp.ones((d, d), dtype=jnp.float32) / (d * d)

    print(f"    T (gradient field) shape: {T.shape}")
    print(f"    ψ (auxiliary potential) shape: {psi.shape}")
    print(f"    ρ_F (fixed measure) shape: {rho_F.shape}")

    # ========================================================================
    # Compute Gradients
    # ========================================================================
    print("\n[5] Computing surrogate gradients...")

    # Compute F gradients
    import time
    start = time.perf_counter()
    Fgrads = solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
    time_F = time.perf_counter() - start

    # Compute p gradient
    start = time.perf_counter()
    pgrad = solver.compute_p_gradient(F, rho_F, psi, verbose=False)
    time_p = time.perf_counter() - start

    print(f"    F gradients: {Fgrads.shape} (computed in {time_F:.4f}s)")
    print(f"    p gradient:  {pgrad.shape} (computed in {time_p:.4f}s)")

    # ========================================================================
    # Analyze Gradients
    # ========================================================================
    print("\n[6] Gradient statistics:")
    for i in range(n_transforms):
        grad_norm = jnp.sqrt(jnp.sum(Fgrads[i]**2))
        print(f"    F[{i}] gradient L2 norm: {grad_norm:.6e}")

    print(f"    p gradient L2 norm: {jnp.linalg.norm(pgrad):.6e}")

    # ========================================================================
    # Simulate Gradient Descent Step
    # ========================================================================
    print("\n[7] Simulating gradient descent update...")

    learning_rate = 0.01

    # Update F (simplified - in practice you'd process gradients more carefully)
    # Note: This is a placeholder. Real F updates need to:
    # 1. Extract gradients w.r.t. matrix elements
    # 2. Ensure affine structure is preserved
    # 3. Apply constraints (contraction, fixed points, etc.)
    print(f"    Learning rate: {learning_rate}")
    print("    (Gradient processing omitted for simplicity)")

    # Update p with projection to simplex
    p_new = p - learning_rate * pgrad
    p_new = jnp.abs(p_new)  # Ensure non-negative
    p_new = p_new / p_new.sum()  # Project to simplex

    print(f"    Old p: {p}")
    print(f"    New p: {p_new}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED")
    print("=" * 70)
    print(f"""
Summary:
  - Solver initialized for {d}x{d} grid
  - Gradients computed for {n_transforms} transformations
  - F gradients: {Fgrads.shape} vector fields
  - p gradient: {pgrad.shape} scalars
  - Total computation time: {time_F + time_p:.4f}s

Next Steps:
  1. Integrate with FixedMeasureSolver for rho_F computation
  2. Integrate with Sinkhorn solver for T and psi
  3. Implement proper F gradient processing
  4. Add convergence criteria and stopping conditions
  5. Run full optimization loop!
""")


if __name__ == "__main__":
    main()
