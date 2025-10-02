# IFS Fixed Measure Solver

High-performance JAX implementation for computing fixed measures of Iterated Function Systems (IFS) using the Banach fixed-point theorem.

## Features

- **Optimized for GPU/TPU**: Leverages JAX for accelerated computation
- **Minimal recompilation**: Designed for optimization loops where IFS parameters change frequently
- **~70x speedup** after initial compilation (cached execution)
- **~8x faster** than naive implementation in optimization scenarios
- **Correctness verified**: Produces identical results to reference implementation

## Performance Highlights

On a typical GPU (tested on NVIDIA):
- **First call**: ~1.2s (includes JIT compilation)
- **Subsequent calls**: ~0.017s (cached, **68x faster**)
- **Optimization loop**: ~0.017s per iteration (varying F, same shape)

Resolution scaling (cached):
- 128×128: 0.007s
- 256×256: 0.009s
- 512×512: 0.013s
- 1024×1024: 0.034s

## Installation

The solver is a standalone package within IFSOpt. Add the parent directory to your Python path:

```python
import sys
sys.path.insert(0, '/path/to/IFSOpt')

from fixed_measure import FixedMeasureSolver
```

## Quick Start

```python
import jax.numpy as jnp
from fixed_measure import FixedMeasureSolver
from ifs_solver.utils import create_sierpinski_ifs, visualize_measure

# Initialize solver with static parameters (do this ONCE)
solver = FixedMeasureSolver(d=512, eps=1e-6)

# Optional: warmup to trigger compilation
solver.warmup(n_transforms=3)

# Get IFS parameters
F, p = create_sierpinski_ifs()

# Solve for fixed measure (fast after warmup!)
mu_fixed = solver.solve(F=F, p=p)

# Visualize
visualize_measure(mu_fixed, title="Sierpinski Triangle")
```

## Usage in Optimization Loops

The solver is designed for scenarios where you repeatedly solve with different F and p:

```python
# Initialize ONCE
solver = FixedMeasureSolver(d=1024, eps=1e-6)
solver.warmup(n_transforms=3)

# Optimization loop
for step in range(num_optimization_steps):
    # Update IFS parameters (your optimization logic)
    F_new = update_parameters(F, gradients)

    # Solve (no recompilation!)
    mu = solver.solve(F=F_new, p=p)

    # Compute loss and update
    loss = compute_loss(mu, target)
    # ... optimization step ...
```

## API Reference

### `FixedMeasureSolver`

Main solver class.

**Constructor:**
```python
FixedMeasureSolver(d, eps=1e-4, max_iterations=1000, min_iterations=100)
```

**Parameters:**
- `d` (int): Grid dimension, must be power of 2 (e.g., 128, 256, 512, 1024)
- `eps` (float): Convergence threshold for Wasserstein-infinity distance
- `max_iterations` (int): Maximum iterations
- `min_iterations` (int): Minimum iterations before checking convergence

**Methods:**

#### `warmup(n_transforms=3, verbose=True)`
Trigger JIT compilation with dummy data. Recommended before optimization loops.

#### `solve(F, p=None, nu=None, verbose=True)`
Solve for the fixed measure.

**Parameters:**
- `F` (List[jnp.ndarray]): List of n transformation matrices (each 3×3)
- `p` (jnp.ndarray, optional): Probability vector (n,), defaults to uniform
- `nu` (jnp.ndarray, optional): Initial measure (d, d), defaults to uniform
- `verbose` (bool): Print convergence info

**Returns:**
- `mu_fixed` (jnp.ndarray): Fixed measure (d, d)

### Utility Functions

```python
from ifs_solver.utils import (
    visualize_measure,
    validate_transformation_matrix,
    create_sierpinski_ifs,
    create_barnsley_fern_ifs
)
```

## Examples

See the `examples/` directory:

- `test_correctness.py`: Verify correctness
- `benchmark_performance.py`: Performance benchmarks
- `compare_implementations.py`: Compare old vs. new
- `optimization_loop_example.py`: Integration example

Run any example:
```bash
python examples/test_correctness.py
```

## Architecture

The implementation separates static and dynamic components:

### Static Components (compile once)
- Grid dimension `d`
- Convergence parameters (`eps`, `max_iterations`)
- Base grid structure
- Computation kernels

### Dynamic Components (change per call)
- Transformation matrices `F`
- Probability vector `p`
- Initial measure `nu`

This design ensures that changing F and p doesn't trigger recompilation, making the solver ideal for optimization loops.

## Integration with Your Code

Since this is part of a larger optimization system, here's how to integrate:

1. **Import the solver** in your main optimization script
2. **Initialize once** at the start with your grid size
3. **Warmup** before the optimization loop
4. **Call `solve()`** repeatedly with updated F and p

The solver will maintain compiled code across calls as long as:
- The number of transformations `n` stays constant
- The shapes remain the same: F is (n, 3, 3), p is (n,)
- The grid dimension `d` doesn't change

## Design Decisions

**Why separate static/dynamic?**
JAX recompiles when function signatures change, but reuses compiled code when only data values change. By extracting static parameters (`d`, `eps`) from the solve loop, we ensure maximum code reuse.

**Why not make it fully differentiable?**
Making the full solve loop differentiable (for gradient-based optimization of F) would require:
- Custom VJP rules for the while_loop
- Implicit differentiation through the fixed point
- Potentially unstable gradients

For most IFS optimization tasks, gradient-free methods (CMA-ES, genetic algorithms) or surrogate models work better.

## License

Part of IFSOpt project.
