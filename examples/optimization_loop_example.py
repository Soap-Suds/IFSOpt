#!/usr/bin/env python3
"""
Example: Using FixedMeasureSolver in an optimization loop.

This demonstrates the intended use case where F and p change during optimization
but the solver reuses compiled code.
"""

import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import jax
import jax.numpy as jnp
import numpy as np
from fixed_measure import FixedMeasureSolver
from ifs_solver.utils import visualize_measure


def example_optimization_loop():
    """
    Simulated optimization loop where we optimize IFS parameters.

    In a real scenario, you might be:
    - Optimizing F to match a target measure
    - Learning p from data
    - Minimizing some loss function
    """
    print("=" * 70)
    print("OPTIMIZATION LOOP EXAMPLE")
    print("=" * 70)
    print()

    # Configuration
    d = 512
    n_transforms = 3
    n_optimization_steps = 10

    # Initialize solver ONCE
    print(f"Initializing solver (d={d})...")
    solver = FixedMeasureSolver(
        d=d,
        eps=1e-6,
        max_iterations=200,
        min_iterations=50
    )
    print()

    # Warmup (optional but recommended)
    print("Warming up solver...")
    solver.warmup(n_transforms=n_transforms, verbose=False)
    print("✓ Warmup complete\n")

    # Initial IFS parameters
    # Starting point: Sierpinski triangle
    F = [
        jnp.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32),
        jnp.array([
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32),
        jnp.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0]
        ], dtype=jnp.float32)
    ]
    p = jnp.array([1/3, 1/3, 1/3], dtype=jnp.float32)

    # ========================================================================
    # Optimization Loop
    # ========================================================================
    print("Running optimization loop...")
    print("-" * 70)

    import time
    times = []

    for step in range(n_optimization_steps):
        # ------------------------------------------------------------------
        # Step 1: Compute fixed measure for current parameters
        # ------------------------------------------------------------------
        start = time.perf_counter()
        mu_current = solver.solve(F=F, p=p, verbose=False)
        solve_time = time.perf_counter() - start
        times.append(solve_time)

        # ------------------------------------------------------------------
        # Step 2: Compute loss (dummy example)
        # ------------------------------------------------------------------
        # In a real scenario, you'd compute loss w.r.t. some target
        # For example: loss = jnp.sum((mu_current - mu_target)**2)
        loss = jnp.sum(mu_current**2)  # Dummy loss

        # ------------------------------------------------------------------
        # Step 3: Update parameters (dummy gradient descent)
        # ------------------------------------------------------------------
        # In a real scenario, you'd use JAX autodiff and a proper optimizer
        # For this example, we just perturb randomly
        learning_rate = 0.001
        F_updated = []
        for f in F:
            # Add small random perturbation
            gradient = jnp.array(np.random.randn(3, 3), dtype=jnp.float32) * 0.01
            f_new = f - learning_rate * gradient
            # Ensure affine structure
            f_new = f_new.at[2, :].set(jnp.array([0.0, 0.0, 1.0]))
            F_updated.append(f_new)

        F = F_updated

        # Also perturb p slightly
        p_grad = jnp.array(np.random.randn(3), dtype=jnp.float32) * 0.01
        p = p - learning_rate * p_grad
        p = jnp.abs(p)  # Keep positive
        p = p / p.sum()  # Normalize

        # ------------------------------------------------------------------
        # Report
        # ------------------------------------------------------------------
        print(f"Step {step+1:2d}: loss={loss:.6f}, solve_time={solve_time:.4f}s")

    print("-" * 70)
    print()

    # ========================================================================
    # Analysis
    # ========================================================================
    print("Performance Analysis:")
    print(f"  First solve:  {times[0]:.4f} s  (includes compilation)")
    print(f"  Average:      {np.mean(times[1:]):.4f} s  (cached)")
    print(f"  Std dev:      {np.std(times[1:]):.4f} s")
    print(f"  Min:          {np.min(times[1:]):.4f} s")
    print(f"  Max:          {np.max(times[1:]):.4f} s")
    print()

    speedup = times[0] / np.mean(times[1:])
    print(f"✓ Speedup after compilation: {speedup:.2f}x")
    print(f"✓ Ready for integration into your optimization pipeline!")
    print()

    # Visualize final result
    print("Generating visualization of final measure...")
    mu_final = solver.solve(F=F, p=p, verbose=False)
    visualize_measure(
        mu_final,
        title=f"Final Measure (after {n_optimization_steps} steps)",
        save_path="/home/spud/IFSOpt/examples/optimization_result.png"
    )


def example_gradient_based_optimization():
    """
    Example showing how to integrate with JAX autodiff for gradient-based optimization.
    """
    print("\n" + "=" * 70)
    print("GRADIENT-BASED OPTIMIZATION EXAMPLE")
    print("=" * 70)
    print()
    print("This example shows the structure for gradient-based optimization.")
    print("Note: For full autodiff through the solver, you'd need to:")
    print("  1. Make the solver differentiable (custom_vjp)")
    print("  2. Or use implicit differentiation")
    print("  3. Or optimize via gradient-free methods (CMA-ES, etc.)")
    print()

    # For now, we'll show the structure without actual gradients
    d = 256
    solver = FixedMeasureSolver(d=d, eps=1e-6)
    solver.warmup(n_transforms=3, verbose=False)

    # Define a loss function (structure only)
    def loss_fn(F, p, target_measure):
        """
        Compute loss between fixed measure of IFS(F, p) and target.

        In practice, you'd either:
        - Use gradient-free optimization (simplex, CMA-ES, genetic algorithms)
        - Implement custom gradients (implicit differentiation)
        - Use a surrogate model for gradients
        """
        mu_fixed = solver.solve(F=F, p=p, verbose=False)
        loss = jnp.sum((mu_fixed - target_measure)**2)
        return loss

    print("Loss function structure defined.")
    print("Integration with optimizers (Optax, scipy.optimize, etc.) follows standard patterns.")
    print()


if __name__ == "__main__":
    # Example 1: Basic optimization loop
    example_optimization_loop()

    # Example 2: Gradient-based structure
    example_gradient_based_optimization()

    print("=" * 70)
    print("EXAMPLES COMPLETED")
    print("=" * 70)
