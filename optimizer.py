#!/usr/bin/env python3
"""
High-level IFS optimizer with warm-starting support.

This orchestrates the optimization loop, managing:
- Fixed measure computation (warm-started)
- Optimal transport solving (warm-started)
- Gradient computation
- Parameter updates
"""

import jax
import jax.numpy as jnp
import functools
from typing import Tuple, Optional, Callable
from fixed_measure import FixedMeasureSolver
from surrogate_gradients import SurrogateGradientSolver


@jax.jit
def compute_gradient_field(potential_flat: jnp.ndarray, d: int) -> jnp.ndarray:
    """Compute gradient of Brenier potential."""
    potential_2d = potential_flat.reshape((d, d))
    grad_y, grad_x = jnp.gradient(potential_2d)
    return jnp.stack([grad_y, grad_x], axis=0)


@functools.partial(jax.jit, static_argnames=['num_iterations', 'd'])
def compute_auxiliary_potential(phi_F: jnp.ndarray, num_iterations: int, d: int) -> jnp.ndarray:
    """Compute auxiliary potential via power series."""
    def T_star_operator(potential):
        potential_2d = potential.reshape((d, d))
        shifted = jnp.roll(potential_2d, shift=1, axis=0)
        operated = (potential_2d + shifted) * 0.45
        return operated.flatten()

    psi_F = jnp.zeros_like(phi_F)
    current_term = phi_F

    for _ in range(num_iterations):
        psi_F += current_term
        current_term = T_star_operator(current_term)

    return psi_F


class IFSOptimizer:
    """High-level IFS optimizer with warm-starting."""

    def __init__(self, d: int, n_transforms: int, eps_fixed: float = 1e-6,
                 eps_ot: float = 0.01, sinkhorn_solver=None, target_measure: Optional[jnp.ndarray] = None):
        """
        Initialize optimizer.

        Args:
            d: Grid dimension
            n_transforms: Number of IFS transformations
            eps_fixed: Fixed measure convergence threshold
            eps_ot: OT epsilon (regularization)
            sinkhorn_solver: Your Sinkhorn solver instance (pass in with warm-start support)
            target_measure: Target measure (d, d) - can be set later
        """
        self.d = d
        self.n_transforms = n_transforms

        # Initialize solvers
        self.fixed_measure_solver = FixedMeasureSolver(d=d, eps=eps_fixed)
        self.gradient_solver = SurrogateGradientSolver(d=d)
        self.sinkhorn_solver = sinkhorn_solver

        # Warmup
        self.fixed_measure_solver.warmup(n_transforms=n_transforms, verbose=False)
        self.gradient_solver.warmup(n_transforms=n_transforms, verbose=False)

        # State for warm-starting
        self.rho_F_prev = None
        self.brenier_potential_prev = None
        self.target_measure = target_measure

    def step(self, F: jnp.ndarray, p: jnp.ndarray,
             learning_rate_F: float = 0.01, learning_rate_p: float = 0.01,
             update_F_fn: Optional[Callable] = None,
             update_p_fn: Optional[Callable] = None,
             verbose: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Single optimization step with warm-starting.

        Args:
            F: Transformation matrices (n, 3, 3)
            p: Probability vector (n,)
            learning_rate_F: Learning rate for F
            learning_rate_p: Learning rate for p
            update_F_fn: Custom F update function (optional)
            update_p_fn: Custom p update function (optional)
            verbose: Print step info

        Returns:
            F_new: Updated transformations
            p_new: Updated probabilities
            ot_cost: OT cost for monitoring
        """
        # Convert F to list for solver
        F_list = [F[i] for i in range(F.shape[0])]

        # 1. Compute fixed measure (warm-started)
        rho_F = self.fixed_measure_solver.solve(
            F=F_list, p=p,
            mu=self.rho_F_prev,  # Warm start!
            verbose=False
        )

        # 2. Solve OT (warm-started if solver supports it)
        # Note: You need to implement warm-starting in your Sinkhorn solver
        ot_output = self.sinkhorn_solver(
            rho_F.ravel(), self.target_measure.ravel(),
            # init=self.brenier_potential_prev  # Uncomment when Sinkhorn supports warm-start
        )

        # 3. Compute potentials
        T = compute_gradient_field(ot_output.f, self.d)
        psi_flat = compute_auxiliary_potential(ot_output.f, num_iterations=20, d=self.d)
        psi = psi_flat.reshape(self.d, self.d)

        # 4. Compute gradients
        Fgrads = self.gradient_solver.compute_F_gradients(F, p, T, rho_F, verbose=False)
        pgrad = self.gradient_solver.compute_p_gradient(F, rho_F, psi, verbose=False)

        # 5. Update parameters
        if update_F_fn is not None:
            F_new = update_F_fn(F, Fgrads, learning_rate_F)
        else:
            # Simple gradient descent (you'll want to customize this)
            F_new = F - learning_rate_F * Fgrads.reshape(F.shape)

        if update_p_fn is not None:
            p_new = update_p_fn(p, pgrad, learning_rate_p)
        else:
            # Gradient descent + simplex projection
            p_new = p - learning_rate_p * pgrad
            p_new = jnp.abs(p_new)
            p_new = p_new / p_new.sum()

        # 6. Save state for next iteration
        self.rho_F_prev = rho_F
        self.brenier_potential_prev = ot_output.f

        if verbose:
            print(f"OT cost: {ot_output.reg_ot_cost:.6f}, p: {p_new}")

        return F_new, p_new, ot_output.reg_ot_cost

    def optimize(self, F_init: jnp.ndarray, p_init: jnp.ndarray,
                 max_steps: int = 100, **step_kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, list]:
        """
        Run full optimization loop.

        Args:
            F_init: Initial transformations
            p_init: Initial probabilities
            max_steps: Maximum optimization steps
            **step_kwargs: Additional arguments for step()

        Returns:
            F_final: Final transformations
            p_final: Final probabilities
            costs: List of OT costs per iteration
        """
        F, p = F_init, p_init
        costs = []

        for step_num in range(max_steps):
            F, p, cost = self.step(F, p, verbose=(step_num % 10 == 0), **step_kwargs)
            costs.append(float(cost))

        return F, p, costs
