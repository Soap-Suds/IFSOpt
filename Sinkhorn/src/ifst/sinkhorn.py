# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License")

from typing import Any, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import fixed_point_loop
from . import geometry as geom_module
from . import initializers as init_lib
from . import problem as prob_module


class SinkhornState(NamedTuple):
  """Holds the state variables used to solve OT with Sinkhorn."""

  potentials: Tuple[jnp.ndarray, ...]
  errors: Optional[jnp.ndarray] = None

  def set(self, **kwargs: Any) -> "SinkhornState":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  @property
  def fu(self) -> jnp.ndarray:
    """The first dual potential or scaling."""
    return self.potentials[0]

  @property
  def gv(self) -> jnp.ndarray:
    """The second dual potential or scaling."""
    return self.potentials[1]


def marginal_error(
    f_u: jnp.ndarray,
    g_v: jnp.ndarray,
    target: jnp.ndarray,
    geom: geom_module.Geometry,
    axis: int = 0,
    norm_error: Sequence[int] = (1,),
    lse_mode: bool = True
) -> jnp.asarray:
  """Output how far Sinkhorn solution is w.r.t target.

  Args:
    f_u: a vector of potentials or scalings for the first marginal.
    g_v: a vector of potentials or scalings for the second marginal.
    target: target marginal.
    geom: Geometry object.
    axis: axis (0 or 1) along which to compute marginal.
    norm_error: (tuple of int) p's to compute p-norm between marginal/target
    lse_mode: whether operating on scalings or potentials

  Returns:
    Array of floats, quantifying difference between target / marginal.
  """
  if lse_mode:
    marginal = geom.marginal_from_potentials(f_u, g_v, axis=axis)
  else:
    marginal = geom.marginal_from_scalings(f_u, g_v, axis=axis)
  norm_error = jnp.asarray(norm_error)
  return jnp.sum(
      jnp.abs(marginal - target) ** norm_error[:, jnp.newaxis], axis=1
  ) ** (1.0 / norm_error)


def compute_kl_reg_cost(
    f: jnp.ndarray, g: jnp.ndarray, ot_prob: prob_module.LinearProblem,
    lse_mode: bool
) -> jnp.ndarray:
  r"""Compute objective of Sinkhorn for OT problem given dual solutions.

  The objective is evaluated for dual solution ``f`` and ``g``, using
  information contained in  ``ot_prob``. The objective is the regularized
  optimal transport cost (i.e. the cost itself plus entropic and unbalanced
  terms). Situations where marginals ``a`` or ``b`` in ``ot_prob`` have zero
  coordinates are reflected in minus infinity entries in their corresponding
  dual potentials. To avoid NaN that may result when multiplying 0's by infinity
  values, ``jnp.where`` is used to cancel these contributions.

  Args:
    f: jnp.ndarray, potential
    g: jnp.ndarray, potential
    ot_prob: linear optimal transport problem.
    lse_mode: bool, whether to compute total mass in lse or kernel mode.

  Returns:
    The regularized transport cost.
  """
  supp_a = ot_prob.a > 0
  supp_b = ot_prob.b > 0
  fa = ot_prob.geom.potential_from_scaling(ot_prob.a)
  div_a = jnp.sum(jnp.where(supp_a, ot_prob.a * (f - fa), 0.0))

  gb = ot_prob.geom.potential_from_scaling(ot_prob.b)
  div_b = jnp.sum(jnp.where(supp_b, ot_prob.b * (g - gb), 0.0))

  if lse_mode:
    total_sum = jnp.sum(ot_prob.geom.marginal_from_potentials(f, g))
  else:
    u = ot_prob.geom.scaling_from_potential(f)
    v = ot_prob.geom.scaling_from_potential(g)
    total_sum = jnp.sum(ot_prob.geom.marginal_from_scalings(u, v))

  return div_a + div_b + ot_prob.epsilon * (
      jnp.sum(ot_prob.a) * jnp.sum(ot_prob.b) - total_sum
  )


class SinkhornOutput(NamedTuple):
  """Holds the output of a Sinkhorn solver applied to a problem.

  Args:
    potentials: list of optimal dual variables, two vector of size
      ``ot.prob.shape[0]`` and ``ot.prob.shape[1]`` returned by Sinkhorn
    errors: vector or errors, along iterations.
    reg_ot_cost: the regularized optimal transport cost.
    ot_prob: stores the definition of the OT problem.
    threshold: convergence threshold used to control the termination of the
      algorithm.
    converged: whether the output corresponds to a solution whose error is
      below the convergence threshold.
    inner_iterations: number of iterations that were run between two
      computations of errors.
  """

  potentials: Tuple[jnp.ndarray, ...]
  errors: Optional[jnp.ndarray] = None
  reg_ot_cost: Optional[jnp.ndarray] = None
  ot_prob: Optional[prob_module.LinearProblem] = None
  threshold: Optional[jnp.ndarray] = None
  converged: Optional[bool] = None
  inner_iterations: Optional[int] = None

  def set(self, **kwargs: Any) -> "SinkhornOutput":
    """Return a copy of self, with potential overwrites."""
    return self._replace(**kwargs)

  @property
  def f(self) -> jnp.ndarray:
    """The first dual potential."""
    return self.potentials[0]

  @property
  def g(self) -> jnp.ndarray:
    """The second dual potential."""
    return self.potentials[1]


@jax.tree_util.register_pytree_node_class
class Sinkhorn:
  r"""Sinkhorn solver.

  The Sinkhorn algorithm is a fixed point iteration that solves a
  regularized optimal transport (reg-OT) problem between two measures.

  Args:
    lse_mode: :obj:`True` for log-sum-exp computations, :obj:`False` for kernel
      multiplication.
    threshold: tolerance used to stop the Sinkhorn iterations.
    norm_error: power used to define the :math:`p`-norm used to quantify
      the magnitude of the gradients.
    inner_iterations: the Sinkhorn error is not recomputed at each
      iteration but every ``inner_iterations`` instead.
    min_iterations: the minimum number of Sinkhorn iterations carried
      out before the error is computed and monitored.
    max_iterations: the maximum number of Sinkhorn iterations.
    initializer: method to compute the initial potentials/scalings.
    num_levels: number of resolution levels for multiscale solver.
      ``num_levels=1`` uses single-scale (default). Higher values use
      coarse-to-fine approach with Grid geometry only.
  """

  def __init__(
      self,
      lse_mode: bool = True,
      threshold: float = 1e-3,
      norm_error: int = 1,
      inner_iterations: int = 10,
      min_iterations: int = 0,
      max_iterations: int = 2000,
      initializer: Optional[init_lib.SinkhornInitializer] = None,
      num_levels: int = 1,
  ):
    self.lse_mode = lse_mode
    self.threshold = threshold
    self.inner_iterations = inner_iterations
    self.min_iterations = min_iterations
    self.max_iterations = max_iterations
    self._norm_error = norm_error
    self.num_levels = num_levels
    self.initializer = init_lib.DefaultInitializer(
    ) if initializer is None else initializer

  def _solve_at_level(
      self,
      ot_prob: prob_module.LinearProblem,
      init: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
      **kwargs: Any,
  ) -> SinkhornOutput:
    """Solve Sinkhorn at a single resolution level.

    Args:
      ot_prob: Linear OT problem.
      init: Initial dual potentials/scalings ``f_u`` and ``g_v``.
        If :obj:`None`, run the initializer.
      kwargs: Keyword arguments for the initializer.

    Returns:
      The Sinkhorn output.
    """
    if init is None:
      if ot_prob.init_f is not None and ot_prob.init_g is not None:
        init = (ot_prob.init_f, ot_prob.init_g)
      else:
        init = self.initializer(ot_prob, lse_mode=self.lse_mode, **kwargs)
    return run(ot_prob, self, init)

  def __call__(
      self,
      ot_prob: prob_module.LinearProblem,
      init: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
      **kwargs: Any,
  ) -> SinkhornOutput:
    """Run Sinkhorn algorithm with optional multiscale approach.

    Args:
      ot_prob: Linear OT problem.
      init: Initial dual potentials/scalings ``f_u`` and ``g_v``.
        If :obj:`None`, run the initializer.
      kwargs: Keyword arguments for the initializer.

    Returns:
      The Sinkhorn output.
    """
    if self.num_levels == 1:
      return self._solve_at_level(ot_prob, init, **kwargs)

    from . import geometry as geom_module

    if not isinstance(ot_prob.geom, geom_module.Grid):
      raise ValueError("Multiscale solver only supports Grid geometry.")

    original_geom = ot_prob.geom
    init_f_coarse = None
    init_g_coarse = None

    for level in range(self.num_levels - 1, -1, -1):
      factor = 2 ** level

      if level > 0:
        coarse_geom = original_geom.coarsen(factor)
        coarse_shape = coarse_geom.grid_size

        a_reshaped = ot_prob.a.reshape(original_geom.grid_size)
        b_reshaped = ot_prob.b.reshape(original_geom.grid_size)

        a_coarse = jax.image.resize(
            a_reshaped, shape=coarse_shape, method='bilinear'
        ).ravel()
        b_coarse = jax.image.resize(
            b_reshaped, shape=coarse_shape, method='bilinear'
        ).ravel()

        a_coarse = a_coarse / jnp.sum(a_coarse)
        b_coarse = b_coarse / jnp.sum(b_coarse)

        level_problem = prob_module.LinearProblem(
            geom=coarse_geom,
            a=a_coarse,
            b=b_coarse,
            init_f=init_f_coarse,
            init_g=init_g_coarse,
            tau_a=ot_prob.tau_a,
            tau_b=ot_prob.tau_b
        )
      else:
        level_problem = prob_module.LinearProblem(
            geom=original_geom,
            a=ot_prob.a,
            b=ot_prob.b,
            init_f=init_f_coarse,
            init_g=init_g_coarse,
            tau_a=ot_prob.tau_a,
            tau_b=ot_prob.tau_b
        )

      output = self._solve_at_level(level_problem, **kwargs)

      if level > 0:
        next_factor = 2 ** (level - 1)
        if next_factor == 1:
          target_shape = original_geom.grid_size
        else:
          target_shape = original_geom.coarsen(next_factor).grid_size

        init_f_coarse = coarse_geom.upsample(output.f, target_shape)
        init_g_coarse = coarse_geom.upsample(output.g, target_shape)

    return output

  def lse_step(
      self, ot_prob: prob_module.LinearProblem, state: SinkhornState,
      iteration: int
  ) -> SinkhornState:
    """Sinkhorn LSE update."""
    old_fu, old_gv = state.fu, state.gv

    new_gv = ot_prob.geom.update_potential(
        old_fu, old_gv, jnp.log(ot_prob.b), iteration, axis=0
    )
    gv = new_gv

    new_fu = ot_prob.geom.update_potential(
        old_fu, gv, jnp.log(ot_prob.a), iteration, axis=1
    )
    fu = new_fu

    return state.set(potentials=(fu, gv))

  def kernel_step(
      self, ot_prob: prob_module.LinearProblem, state: SinkhornState,
      iteration: int
  ) -> SinkhornState:
    """Sinkhorn multiplicative update."""
    old_gv = state.gv
    new_gv = ot_prob.geom.update_scaling(
        state.fu, ot_prob.b, iteration, axis=0
    )
    gv = new_gv
    new_fu = ot_prob.geom.update_scaling(
        gv,
        ot_prob.a,
        iteration,
        axis=1
    )
    fu = new_fu
    return state.set(potentials=(fu, gv))

  def one_iteration(
      self, ot_prob: prob_module.LinearProblem, state: SinkhornState,
      iteration: int, compute_error: bool
  ) -> SinkhornState:
    """Carries out one Sinkhorn iteration.

    Args:
      ot_prob: the transport problem definition
      state: SinkhornState named tuple.
      iteration: the current iteration of the Sinkhorn loop.
      compute_error: flag to indicate this iteration computes/stores an error

    Returns:
      The updated state.
    """
    if self.lse_mode:
      state = self.lse_step(ot_prob, state, iteration)
    else:
      state = self.kernel_step(ot_prob, state, iteration)

    err = jax.lax.cond(
        jnp.logical_or(
            iteration == self.max_iterations - 1,
            jnp.logical_and(compute_error, iteration >= self.min_iterations)
        ),
        lambda state, prob: marginal_error(
            state.fu, state.gv, prob.b, prob.geom, 0, self.norm_error,
            self.lse_mode
        )[0],
        lambda *_: jnp.array(jnp.inf, dtype=ot_prob.dtype),
        state,
        ot_prob,
    )
    errors = state.errors.at[iteration // self.inner_iterations, :].set(err)
    state = state.set(errors=errors)

    return state

  def _converged(self, state: SinkhornState, iteration: int) -> bool:
    err = state.errors[iteration // self.inner_iterations - 1, 0]
    return jnp.logical_and(iteration > 0, err < self.threshold)

  def _diverged(self, state: SinkhornState, iteration: int) -> bool:
    err = state.errors[iteration // self.inner_iterations - 1, 0]
    return jnp.logical_not(jnp.isfinite(err))

  def _continue(self, state: SinkhornState, iteration: int) -> bool:
    """Continue while not(converged) and not(diverged)."""
    return jnp.logical_and(
        jnp.logical_not(self._diverged(state, iteration)),
        jnp.logical_not(self._converged(state, iteration))
    )

  @property
  def outer_iterations(self) -> int:
    """Upper bound on number of times inner_iterations are carried out."""
    return np.ceil(self.max_iterations / self.inner_iterations).astype(int)

  def init_state(
      self, ot_prob: prob_module.LinearProblem, init: Tuple[jnp.ndarray,
                                                           jnp.ndarray]
  ) -> SinkhornState:
    """Return the initial state of the loop."""
    errors = -jnp.ones((self.outer_iterations, len(self.norm_error)),
                       dtype=ot_prob.dtype)
    state = SinkhornState(init, errors=errors)
    return state

  def output_from_state(
      self, ot_prob: prob_module.LinearProblem, state: SinkhornState
  ) -> SinkhornOutput:
    """Create an output from a loop state.

    Args:
      ot_prob: the transport problem.
      state: a SinkhornState.

    Returns:
      A SinkhornOutput.
    """
    geom = ot_prob.geom

    f = state.fu if self.lse_mode else geom.potential_from_scaling(state.fu)
    g = state.gv if self.lse_mode else geom.potential_from_scaling(state.gv)

    converged = jnp.logical_and(
        jnp.logical_not(jnp.any(jnp.isnan(state.errors))), state.errors[-1]
        < self.threshold
    )[0]

    reg_ot_cost = compute_kl_reg_cost(f, g, ot_prob, self.lse_mode)

    return SinkhornOutput((f, g),
                          errors=state.errors[:, 0],
                          reg_ot_cost=reg_ot_cost,
                          ot_prob=ot_prob,
                          threshold=jnp.array(self.threshold),
                          converged=converged,
                          inner_iterations=self.inner_iterations)

  @property
  def norm_error(self) -> Tuple[int, ...]:
    """Powers used to compute the p-norm between marginal/target."""
    return self._norm_error,

  def tree_flatten(self):
    aux = vars(self).copy()
    aux["norm_error"] = aux.pop("_norm_error")
    aux.pop("threshold")
    return [self.threshold], aux

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(**aux_data, threshold=children[0])


def run(
    ot_prob: prob_module.LinearProblem, solver: Sinkhorn,
    init: Tuple[jnp.ndarray, ...]
) -> SinkhornOutput:
  """Run loop of the solver, outputting a state upgraded to an output."""
  out = iterations(ot_prob, solver, init)
  return out.set(ot_prob=ot_prob)


def iterations(
    ot_prob: prob_module.LinearProblem, solver: Sinkhorn,
    init: Tuple[jnp.ndarray, ...]
) -> SinkhornOutput:
  """Jittable Sinkhorn loop."""

  def cond_fn(
      iteration: int, const: Tuple[prob_module.LinearProblem, Sinkhorn],
      state: SinkhornState
  ) -> bool:
    _, solver = const
    return solver._continue(state, iteration)

  def body_fn(
      iteration: int, const: Tuple[prob_module.LinearProblem, Sinkhorn],
      state: SinkhornState, compute_error: bool
  ) -> SinkhornState:
    ot_prob, solver = const
    return solver.one_iteration(ot_prob, state, iteration, compute_error)

  const = ot_prob, solver
  state = solver.init_state(ot_prob, init)
  state = fixed_point_loop.fixpoint_iter(
      cond_fn, body_fn, solver.min_iterations, solver.max_iterations,
      solver.inner_iterations, const, state
  )
  return solver.output_from_state(ot_prob, state)