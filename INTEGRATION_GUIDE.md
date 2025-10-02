# Integration Guide for IFS Solver

## Quick Integration Checklist

- [x] Solver implemented and optimized (7.9x faster than baseline)
- [x] Correctness verified (output matches reference within 1e-10)
- [x] Performance benchmarked (68x speedup after compilation)
- [x] Ready for optimization loops

## How to Use in Your Optimization Code

### Step 1: Import

At the top of your optimization script:

```python
import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import jax
import jax.numpy as jnp
from fixed_measure import FixedMeasureSolver
```

### Step 2: Initialize (Once per optimization run)

```python
# Configuration
GRID_SIZE = 1024  # Power of 2: 128, 256, 512, 1024, 2048
CONVERGENCE_THRESHOLD = 1e-6
N_TRANSFORMS = 3  # Number of IFS maps

# Create solver
solver = FixedMeasureSolver(
    d=GRID_SIZE,
    eps=CONVERGENCE_THRESHOLD,
    max_iterations=300,
    min_iterations=50
)

# Warmup (optional but recommended)
print("Warming up solver...")
solver.warmup(n_transforms=N_TRANSFORMS)
print("Ready!")
```

### Step 3: Use in Optimization Loop

```python
# Initialize your IFS parameters
F = [...]  # List of n transformation matrices (3x3 JAX arrays)
p = jnp.array([...])  # Probability vector (n,)

# Optimization loop
for iteration in range(num_iterations):
    # 1. Compute fixed measure for current F, p
    mu_current = solver.solve(F=F, p=p, verbose=False)

    # 2. Compute your loss/objective
    loss = your_loss_function(mu_current, target_data)

    # 3. Update F and p using your optimization algorithm
    # (gradient-free methods recommended: CMA-ES, genetic algorithms, etc.)
    F, p = your_optimizer.step(F, p, loss)

    # 4. Log progress
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: loss = {loss:.6f}")
```

## Performance Expectations

### Timing Breakdown (d=512, GPU)

| Phase | Time | Notes |
|-------|------|-------|
| `solver.warmup()` | ~1.2s | One-time cost |
| First `solve()` call | ~0.02s | Already warmed up |
| Subsequent `solve()` | ~0.017s | Fully cached |

### Scaling with Grid Size

| Grid Size | Time per solve | Memory |
|-----------|----------------|--------|
| 512×512 | 0.013s | ~1 MB |
| 1024×1024 | 0.034s | ~4 MB |
| 2048×2048 | 0.120s | ~16 MB |

**Recommendation**: Start with 512×512 for fast iteration, use 1024×1024 for final results.

## Common Integration Patterns

### Pattern 1: Gradient-Free Optimization

```python
from scipy.optimize import minimize

solver = FixedMeasureSolver(d=512, eps=1e-6)
solver.warmup(n_transforms=3)

def objective(params):
    """Convert flat parameter vector to F, p and evaluate."""
    F, p = params_to_ifs(params)
    mu = solver.solve(F=F, p=p, verbose=False)
    return compute_loss(mu, target)

# Use your favorite optimizer
result = minimize(
    objective,
    x0=initial_params,
    method='Nelder-Mead',  # or CMA-ES, BFGS, etc.
    options={'maxiter': 1000}
)
```

### Pattern 2: Custom Optimization Loop

```python
solver = FixedMeasureSolver(d=512, eps=1e-6)
solver.warmup(n_transforms=3)

# Initialize
F_current = initialize_ifs()
p_current = jnp.ones(3) / 3

# Your custom algorithm
for step in range(max_steps):
    # Evaluate
    mu = solver.solve(F=F_current, p=p_current, verbose=False)
    loss = compute_loss(mu)

    # Update (your custom logic)
    F_next, p_next = your_update_rule(F_current, p_current, loss)

    # Accept/reject (if using simulated annealing, genetic algorithm, etc.)
    if acceptance_criterion(loss_new, loss_old):
        F_current, p_current = F_next, p_next
```

### Pattern 3: Batch Evaluation (Future)

For evaluating multiple IFS candidates in parallel:

```python
# Current: Sequential (simple)
results = []
for F_candidate in population:
    mu = solver.solve(F=F_candidate, p=p)
    results.append(mu)

# Future: Could batch with vmap
# This would require refactoring solver.solve() to be vmappable
```

## Important Constraints

### What Triggers Recompilation?

**Safe (no recompilation):**
- Changing values in F (as long as shape stays (n, 3, 3))
- Changing values in p (as long as shape stays (n,))
- Changing nu (initial measure)

**Triggers recompilation:**
- Changing number of transforms n
- Changing grid size d
- Changing convergence parameters (eps, max_iterations)

### Best Practices

1. **Fix n early**: Decide on number of transforms before optimization
2. **Reuse solver instance**: Don't create new `FixedMeasureSolver()` in loop
3. **Warmup once**: Call `warmup()` before loop, not inside
4. **Batch logging**: Use `verbose=False` in solve, log separately

## Example: Full Optimization Script

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/spud/IFSOpt')

import jax.numpy as jnp
from fixed_measure import FixedMeasureSolver

def optimize_ifs(target_measure, num_steps=100):
    """
    Optimize IFS to match a target measure.

    Args:
        target_measure: Target distribution (d, d)
        num_steps: Number of optimization steps

    Returns:
        Optimized F, p, and final measure
    """
    d = target_measure.shape[0]

    # Setup
    solver = FixedMeasureSolver(d=d, eps=1e-6)
    solver.warmup(n_transforms=3)

    # Initialize (example: random perturbation of identity)
    F = initialize_random_ifs(n=3)
    p = jnp.ones(3) / 3

    # Optimization
    best_loss = float('inf')
    best_F, best_p = F, p

    for step in range(num_steps):
        # Evaluate
        mu = solver.solve(F=F, p=p, verbose=False)
        loss = jnp.sum((mu - target_measure)**2)

        # Track best
        if loss < best_loss:
            best_loss = loss
            best_F, best_p = F, p

        # Update (simple gradient-free: random perturbation + accept better)
        F_candidate = perturb_ifs(F, noise_scale=0.01)
        p_candidate = perturb_probabilities(p, noise_scale=0.01)

        mu_candidate = solver.solve(F=F_candidate, p=p_candidate, verbose=False)
        loss_candidate = jnp.sum((mu_candidate - target_measure)**2)

        if loss_candidate < loss:
            F, p = F_candidate, p_candidate

        if step % 10 == 0:
            print(f"Step {step}: loss={best_loss:.6f}")

    return best_F, best_p, solver.solve(F=best_F, p=best_p)

# Run
target = create_target_measure()
F_opt, p_opt, mu_opt = optimize_ifs(target, num_steps=100)
```

## Troubleshooting

### "Solver is slow"
- Make sure you called `warmup()` before the loop
- Check you're reusing the same solver instance
- Verify you're on GPU: `print(jax.default_backend())`

### "Results don't match expected"
- Run `examples/test_correctness.py` to verify installation
- Check convergence: increase `max_iterations` or decrease `eps`
- Validate your F matrices: use `utils.validate_transformation_matrix()`

### "Memory issues"
- Reduce grid size: try d=512 instead of d=1024
- If using multiple solvers: they can share computation on same device

### "Want gradients through solver"
- Not currently supported (would need custom VJP)
- Use gradient-free optimization methods instead
- Or finite differences for approximate gradients

## Next Steps

1. **Test integration**: Copy one of the patterns above
2. **Benchmark**: Time your optimization loop
3. **Iterate**: Tune convergence parameters for your problem
4. **Scale**: Once working, increase grid size for better resolution

## Questions?

- Code: See `fixed_measure/core.py` for implementation details
- Examples: Check `examples/optimization_loop_example.py`
- Performance: Review `examples/benchmark_performance.py`
