"""Example: Compute Sinkhorn distance between two images using Grid geometry.

This script demonstrates the use of the ifst package to compute the
entropic-regularized Sinkhorn distance between two 2D images (probability
distributions on grids) using JAX for GPU acceleration.
"""

import sys
sys.path.insert(0, '../src')

import time
import jax
import jax.numpy as jnp
from ifst import Grid, LinearProblem, Sinkhorn


def main():
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)

    # Create two simple 64x64 images
    size = 1024

    # Create first image: Gaussian blob centered at (20, 20)
    x = jnp.linspace(0, 1, size)
    y = jnp.linspace(0, 1, size)
    X, Y = jnp.meshgrid(x, y)

    # Image 1: Gaussian centered at (0.3, 0.3)
    image1 = jnp.exp(-100 * ((X - 0.3)**2 + (Y - 0.3)**2))

    # Image 2: Gaussian centered at (0.7, 0.7)
    image2 = jnp.exp(-100 * ((X - 0.7)**2 + (Y - 0.7)**2))

    # Normalize to probability distributions
    image1 = image1 / jnp.sum(image1)
    image2 = image2 / jnp.sum(image2)

    # Flatten the images
    a = image1.ravel()
    b = image2.ravel()

    print(f"Image size: {size}x{size}")
    print(f"Number of pixels: {size * size}")
    print(f"Sum of image1: {jnp.sum(a):.6f}")
    print(f"Sum of image2: {jnp.sum(b):.6f}")

    # Create Grid geometry from the images
    # The grid is defined by the spatial locations in the images
    grid_geom = Grid(
        grid_size=(size, size),
        epsilon=0.01,  # Regularization parameter
    )

    print(f"\nGrid geometry created:")
    print(f"  Grid size: {grid_geom.grid_size}")
    print(f"  Grid dimension: {grid_geom.grid_dimension}")
    print(f"  Total points: {grid_geom.num_a}")
    print(f"  Epsilon: {grid_geom.epsilon}")

    # Create the linear OT problem
    ot_problem = LinearProblem(
        geom=grid_geom,
        a=a,
        b=b
    )

    print(f"\nLinear OT problem created:")
    print(f"  Problem is balanced: {ot_problem.is_balanced}")
    print(f"  Problem is uniform: {ot_problem.is_uniform}")

    # Create and run the Sinkhorn solver
    solver = Sinkhorn(
        threshold=1e-3,
        max_iterations=1000,
        lse_mode=True  # Use log-sum-exp mode for stability
    )

    print(f"\nRunning Sinkhorn algorithm...")
    print(f"  Max iterations: {solver.max_iterations}")
    print(f"  Threshold: {solver.threshold}")
    print(f"  LSE mode: {solver.lse_mode}")

    # Solve the problem
    start_time = time.time()
    output = solver(ot_problem)
    output.f.block_until_ready()
    elapsed_time = time.time() - start_time

    # Print results
    print(f"\nResults:")
    print(f"  Converged: {output.converged}")
    print(f"  Final error: {output.errors[-1]:.6e}")
    print(f"  Sinkhorn distance (regularized OT cost): {output.reg_ot_cost:.6f}")
    print(f"  Number of iterations: {jnp.sum(output.errors != -1) * output.inner_iterations}")
    print(f"  Time elapsed: {elapsed_time:.3f} seconds")

    # Additional information about the dual potentials
    print(f"\nDual potentials:")
    print(f"  f (first potential) shape: {output.f.shape}")
    print(f"  g (second potential) shape: {output.g.shape}")
    print(f"  f min/max: {jnp.min(output.f[jnp.isfinite(output.f)]):.3f} / {jnp.max(output.f[jnp.isfinite(output.f)]):.3f}")
    print(f"  g min/max: {jnp.min(output.g[jnp.isfinite(output.g)]):.3f} / {jnp.max(output.g[jnp.isfinite(output.g)]):.3f}")

    # Demonstrate warm-start capability
    print(f"\n{'='*60}")
    print("Demonstrating Warm-Start Capability")
    print(f"{'='*60}")

    # Create random initial potentials for warm-start
    key, subkey1, subkey2 = jax.random.split(key, 3)
    f_init = jax.random.normal(subkey1, shape=(size * size,)) * 0.1
    g_init = jax.random.normal(subkey2, shape=(size * size,)) * 0.1

    print(f"\nCreated random initial potentials:")
    print(f"  f_init shape: {f_init.shape}")
    print(f"  g_init shape: {g_init.shape}")
    print(f"  f_init mean/std: {jnp.mean(f_init):.3f} / {jnp.std(f_init):.3f}")
    print(f"  g_init mean/std: {jnp.mean(g_init):.3f} / {jnp.std(g_init):.3f}")

    # Create a new LinearProblem with warm-start potentials
    ot_problem_warmstart = LinearProblem(
        geom=grid_geom,
        a=a,
        b=b,
        init_f=f_init,
        init_g=g_init
    )

    print(f"\nRunning Sinkhorn with warm-start...")

    # Run the Sinkhorn solver with warm-start
    start_time_warmstart = time.time()
    output_warmstart = solver(ot_problem_warmstart)
    output_warmstart.f.block_until_ready()
    elapsed_time_warmstart = time.time() - start_time_warmstart

    # Print warm-start results
    print(f"\nWarm-start Results:")
    print(f"  Converged: {output_warmstart.converged}")
    print(f"  Final error: {output_warmstart.errors[-1]:.6e}")
    print(f"  Warm-started transport cost: {output_warmstart.reg_ot_cost:.6f}")
    print(f"  Number of iterations: {jnp.sum(output_warmstart.errors != -1) * output_warmstart.inner_iterations}")
    print(f"  Time elapsed: {elapsed_time_warmstart:.3f} seconds")

    print(f"\nComparison:")
    print(f"  Standard initialization cost: {output.reg_ot_cost:.6f}")
    print(f"  Warm-start initialization cost: {output_warmstart.reg_ot_cost:.6f}")
    print(f"  Cost difference: {abs(output.reg_ot_cost - output_warmstart.reg_ot_cost):.6e}")
    print(f"  Standard time: {elapsed_time:.3f}s")
    print(f"  Warm-start time: {elapsed_time_warmstart:.3f}s")
    print(f"  Time difference: {elapsed_time - elapsed_time_warmstart:.3f}s")

    # Demonstrate multiscale capability
    print(f"\n{'='*60}")
    print("Demonstrating Multiscale (Coarse-to-Fine) Solver")
    print(f"{'='*60}")

    # Create a multiscale solver with 4 levels
    solver_multiscale = Sinkhorn(
        threshold=1e-3,
        max_iterations=1000,
        lse_mode=True,
        num_levels=4
    )

    print(f"\nMultiscale solver configuration:")
    print(f"  Number of levels: {solver_multiscale.num_levels}")
    print(f"  Resolution hierarchy: ", end="")
    for level in range(solver_multiscale.num_levels - 1, -1, -1):
        factor = 2 ** level
        coarse_size = size // factor
        print(f"{coarse_size}x{coarse_size}", end="")
        if level > 0:
            print(" → ", end="")
    print()

    print(f"\nRunning multiscale Sinkhorn...")

    # Run the multiscale solver
    start_time_multiscale = time.time()
    output_multiscale = solver_multiscale(ot_problem)
    output_multiscale.f.block_until_ready()
    elapsed_time_multiscale = time.time() - start_time_multiscale

    # Print multiscale results
    print(f"\nMultiscale Results:")
    print(f"  Converged: {output_multiscale.converged}")
    print(f"  Final error: {output_multiscale.errors[-1]:.6e}")
    print(f"  Multiscale transport cost: {output_multiscale.reg_ot_cost:.6f}")
    print(f"  Number of iterations (final level): {jnp.sum(output_multiscale.errors != -1) * output_multiscale.inner_iterations}")
    print(f"  Time elapsed: {elapsed_time_multiscale:.3f} seconds")

    print(f"\nPerformance Comparison:")
    print(f"  Single-scale time: {elapsed_time:.3f}s")
    print(f"  Multiscale time: {elapsed_time_multiscale:.3f}s")
    print(f"  Speedup: {elapsed_time / elapsed_time_multiscale:.2f}x")
    print(f"  Cost difference: {abs(output.reg_ot_cost - output_multiscale.reg_ot_cost):.6e}")


if __name__ == "__main__":
    main()